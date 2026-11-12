//! Qwen3-ASR's audio tower on the device, as projections of gathered rows: `Qwen3ASREncoder`.
//!
//! - A clip is mel frames, 128 BF16 features a row, in chunks of 100 frames that the convolutions
//!   see alone. A launch holds [`CHUNKS`] chunks; its rows past a clip's are computed and read by
//!   no caller.
//! - `conv2d1` sees one input channel, so each output time's features at every frequency are one
//!   projection of its three input frames: a band of the kernel's taps per frequency, three
//!   projections of 128 features into 64 frequencies of 512 channels, summed.
//! - `conv2d2` and `conv2d3` are nine projections each, one a kernel tap, of the rows the host
//!   names: a row is one (time, frequency) pixel's 512 channels, the 480 the checkpoint stores and
//!   zeros, `Channels` into `Filters` and back. A tap outside the image reads a zero row the caller
//!   writes.
//! - The third convolution's rows are time-major, so a time's 16 frequencies are one row of 8192
//!   features, which `conv_out` projects, in the checkpoint's channel-major order rearranged.
//! - Each output time adds its position's sinusoid, its index within its chunk.

use furiosa_opt_std::prelude::*;

use std::marker::PhantomData;

pub use crate::ops::attention::single::Rows;

use serde::Deserialize;

use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::models::config::{self, Constant, Literal};
use crate::models::{Class, Implements};
use crate::ops::attention::Attention;
use crate::ops::attention::single::{Head, Tokens, Trips, Visible};
use crate::ops::linear::{Affine, Linear, PerHead};
use crate::ops::normalization::LayerNorm;
use crate::tensor::parallel::Chip;
use crate::tensor::scalar::F32;
use crate::{Module, Op};

axes![
    Mel = 128,
    Band = 32768,
    Channels = 512,
    Filters = 512,
    Flat = 8192,
    Width = 1024,
    Phases = 16
];

/// Chunks of 100 mel frames a launch holds: 8 s of audio, one attention window.
pub const CHUNKS: usize = 8;

/// Mel frames of a chunk.
pub const FRAMES: usize = 100;

/// Output times of a chunk after each convolution, and output frequencies.
pub const TIMES: [usize; 3] = [50, 25, 13];
pub const FREQUENCIES: [usize; 3] = [64, 32, 16];

/// Channels the checkpoint stores of each convolution's output.
const STORED: usize = 480;

/// BF16 coefficients over `In` with an FP32 bias of `Out`.
pub type Biased<In, Out> =
    Affine<HbmTensor<bf16, Chip, m![Out, In]>, HbmTensor<f32, Chip, m![Out]>>;

/// The tower's convolutions, its flattening projection and its position table, as the device
/// projects them: each convolution a projection a kernel tap, the bias in the first.
#[derive(DeviceSend)]
pub struct Frontend {
    pub conv1: [Biased<Mel, Band>; 3],
    pub conv2: [Biased<Channels, Filters>; 9],
    pub conv3: [Biased<Filters, Channels>; 9],
    pub conv_out: Linear<Flat, Width, Chip>,
    pub positions: HbmTensor<bf16, Chip, m![Phases, Width]>,
}

/// The convolutions under `conv2d1`..`conv2d3`, the projection under `conv_out` and the sinusoids
/// of the chunk's times, each rearranged on host.
impl<'a> Load<&'a Checkpoint> for Frontend {
    fn load(cx: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        let floats = |cx: &mut Context<'_>, name: &str, part: &str, shape: &[usize]| {
            cx.field(name, name, |cx| {
                cx.field(part, part, |cx| cx.floats(from, shape))
            })
        };
        let w1 = floats(cx, "conv2d1", "weight", &[STORED, 1, 3, 3])?;
        let b1 = floats(cx, "conv2d1", "bias", &[STORED])?;
        let conv1 = taps::<3, Mel, Band>(
            cx,
            |tap, out, input| {
                // Output feature `out` is frequency `out / 512`, channel `out % 512`; input feature
                // `input` is frequency `input`.
                let (frequency, channel) = (out / Channels::SIZE, out % Channels::SIZE);
                let row = (2 * frequency + 1).checked_sub(input)?;
                (channel < STORED && row < 3).then(|| w1[(channel * 3 + (2 - row)) * 3 + tap])
            },
            |out| {
                let channel = out % Channels::SIZE;
                (channel < STORED).then(|| b1[channel])
            },
        )?;
        let square = |cx: &mut Context<'_>, name: &str| {
            let weight = floats(cx, name, "weight", &[STORED, STORED, 3, 3])?;
            let bias = floats(cx, name, "bias", &[STORED])?;
            let coefficient = move |tap: usize, out: usize, input: usize| {
                (out < STORED && input < STORED).then(|| weight[(out * STORED + input) * 9 + tap])
            };
            let bias = move |out: usize| (out < STORED).then(|| bias[out]);
            Ok::<_, LoadError>((coefficient, bias))
        };
        let (weight, bias) = square(cx, "conv2d2")?;
        let conv2 = taps::<9, Channels, Filters>(cx, &weight, &bias)?;
        let (weight, bias) = square(cx, "conv2d3")?;
        let conv3 = taps::<9, Filters, Channels>(cx, &weight, &bias)?;
        let frequencies = FREQUENCIES[2];
        let stored = floats(
            cx,
            "conv_out",
            "weight",
            &[Width::SIZE, STORED * frequencies],
        )?;
        let flat: Vec<bf16> = (0..Width::SIZE * Flat::SIZE)
            .map(|at| {
                let (out, input) = (at / Flat::SIZE, at % Flat::SIZE);
                let (frequency, channel) = (input / Channels::SIZE, input % Channels::SIZE);
                let value = match channel < STORED {
                    true => stored[out * STORED * frequencies + channel * frequencies + frequency],
                    false => 0.0,
                };
                bf16::from_f32(value)
            })
            .collect();
        let conv_out = cx.field("conv_out", "conv_out", |cx| {
            Linear::load(
                cx,
                HostTensor::<bf16, m![{ Chip }, { m![Width, Flat] }]>::from_vec(flat),
            )
        })?;
        // Transformers' `SinusoidsPositionEmbedding` of 13 times and 1024 channels, rounded once.
        let half = Width::SIZE / 2;
        let increment = 10000f64.ln() / (half - 1) as f64;
        let positions: Vec<bf16> = (0..Phases::SIZE * Width::SIZE)
            .map(|at| {
                let (time, channel) = (at / Width::SIZE, at % Width::SIZE);
                let inverse = (-increment * (channel % half) as f64).exp() as f32;
                let scaled = time as f32 * inverse;
                bf16::from_f32(if channel < half {
                    scaled.sin()
                } else {
                    scaled.cos()
                })
            })
            .collect();
        let positions = cx.field("positions", "", |cx| {
            HbmTensor::load(
                cx,
                HostTensor::<bf16, m![Phases, Width]>::from_vec(positions),
            )
        })?;
        Ok(Self {
            conv1,
            conv2,
            conv3,
            conv_out,
            positions,
        })
    }
}

/// `Qwen3ASRAudioEncoderLayer`: pre-norm unmasked self attention of 16 heads of 64 features,
/// every projection biased, then a pre-norm GELU feed-forward, each added to its input.
#[derive(DeviceSend, Load)]
pub struct Layer {
    pub self_attn_layer_norm: Norm,
    pub self_attn: SelfAttention,
    pub final_layer_norm: Norm,
    pub fc1: Biased<Width, Intermediate>,
    pub fc2: Biased<Intermediate, Width>,
}

/// The layer's attention: its query, key and value heads, head-major, and its output projection.
#[derive(DeviceSend, Load)]
pub struct SelfAttention {
    pub q_proj: Heads,
    pub k_proj: Heads,
    pub v_proj: Heads,
    pub out_proj: Biased<Inner, Width>,
}

/// A biased projection into head-major heads of 64 features.
pub type Heads = PerHead<Width, Inner, Head, Chip>;

/// `nn.LayerNorm` of BF16 rows, epsilon 1e-5.
pub type Norm = LayerNorm<HbmTensor<bf16, Chip, m![Width]>, HbmTensor<f32, Chip, m![1]>>;

axes![
    Inner = 1024,
    Intermediate = 4096,
    Projected = 1024,
    Text = 2048
];

/// Encoder layers of the tower.
pub const LAYERS: usize = 24;

/// The tower and its projector: the frontend, the layers and the final norm under
/// `model.audio_tower`, then `model.multi_modal_projector`'s GELU pair into the text model's width.
#[derive(DeviceSend)]
pub struct Tower {
    pub frontend: Frontend,
    pub layers: [Layer; LAYERS],
    pub ln_post: Norm,
    pub linear_1: Biased<Width, Projected>,
    pub linear_2: Biased<Projected, Text>,
}

/// The tower under `audio_tower` and the projector under `multi_modal_projector`, below `model`,
/// the text model ignored.
impl<'a> Load<&'a Checkpoint> for Tower {
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.ignore("language_model");
            Self::read(cx, from)
        })
    }
}

impl Tower {
    /// The tower and projector at the current name, `model`.
    fn read(cx: &mut Context<'_>, from: &Checkpoint) -> Result<Self, LoadError> {
        let (frontend, layers, ln_post) = cx.field("", "audio_tower", |cx| {
            let frontend = Frontend::load(cx, from)?;
            let layers = cx.field("layers", "layers", |cx| {
                let mut layers = Vec::with_capacity(LAYERS);
                for at in 0..LAYERS {
                    let name = at.to_string();
                    layers.push(cx.field(&name, &name, |cx| Layer::load(cx, from))?);
                }
                Ok::<_, LoadError>(layers)
            })?;
            let ln_post = cx.field("ln_post", "ln_post", |cx| Norm::load(cx, from))?;
            Ok::<_, LoadError>((frontend, layers, ln_post))
        })?;
        let (linear_1, linear_2) = cx.field("", "multi_modal_projector", |cx| {
            let first = cx.field("linear_1", "linear_1", |cx| Biased::load(cx, from))?;
            let second = cx.field("linear_2", "linear_2", |cx| Biased::load(cx, from))?;
            Ok::<_, LoadError>((first, second))
        })?;
        Ok(Self {
            frontend,
            layers: layers
                .try_into()
                .unwrap_or_else(|_| unreachable!("one layer an index")),
            ln_post,
            linear_1,
            linear_2,
        })
    }
}

/// The tower reads a checkpoint under `model` when its audio config declares the widths, depth,
/// windows and activation the tower compiles, and no load quantizes it.
impl Implements for Tower {
    const SIZE: &'static str = "1.7B";
    const CLASS: Class = Class {
        family: "qwen3_asr",
        name: "Encoder",
    };

    fn scope(source: &Checkpoint) -> Result<&'static str, LoadError> {
        if source.quantization().is_some() {
            return Err(LoadError::Unsupported {
                name: "quantization".into(),
                reason: "the audio tower's projections stay BF16".into(),
            });
        }
        let parse = || {
            let Hf {
                audio_config: audio,
                text_config:
                    TextHf {
                        hidden_size: config::Size { .. },
                    },
            } = config::parse(source.config())?;
            let AudioHf {
                d_model: config::Size { .. },
                encoder_layers: Constant,
                encoder_attention_heads: Constant,
                encoder_ffn_dim: config::Size { .. },
                num_mel_bins: config::Size { .. },
                n_window: Constant,
                n_window_infer: Constant,
                downsample_hidden_size: Constant,
                output_dim: config::Size { .. },
                activation_function: Literal,
            } = audio;
            Ok::<_, std::io::Error>(())
        };
        parse().map(|()| "model").map_err(|error| {
            let (name, reason) = ("config.json".into(), error.to_string());
            match error.kind() {
                std::io::ErrorKind::Unsupported => LoadError::Unsupported { name, reason },
                _ => LoadError::InvalidData { name, reason },
            }
        })
    }
}

/// The config values the tower is compiled for.
#[derive(Deserialize)]
struct Hf {
    audio_config: AudioHf,
    text_config: TextHf,
}

#[derive(Deserialize)]
struct AudioHf {
    d_model: config::Size<Width>,
    encoder_layers: Constant<LAYERS>,
    encoder_attention_heads: Constant<16>,
    encoder_ffn_dim: config::Size<Intermediate>,
    num_mel_bins: config::Size<Mel>,
    n_window: Constant<{ FRAMES / 2 }>,
    n_window_infer: Constant<{ CHUNKS * FRAMES }>,
    downsample_hidden_size: Constant<STORED>,
    output_dim: config::Size<Text>,
    activation_function: Literal<"gelu">,
}

#[derive(Deserialize)]
struct TextHf {
    hidden_size: config::Size<Text>,
}

/// Bidirectional attention of head-major heads over the visible keys, scaled by `1 / 8`.
#[derive(Default, DeviceSend)]
pub struct Bidirectional;

/// The FP32 bits of `1 / sqrt(64)`.
const EIGHTH: usize = 0.125_f32.to_bits() as usize;

/// The heads of a tile of rows: queries, keys, values or the attended values.
pub type Tile = Tokens<Trips<Inner>, Chip>;

impl<'x, Env> Module<(&'x Tile, &'x Tile, &'x Tile, &'x Visible<Chip>), Env, &mut ()>
    for Bidirectional
{
    type Output = Tile;

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        _: &mut (),
        (query, key, value, visible): (&'x Tile, &'x Tile, &'x Tile, &'x Visible<Chip>),
    ) -> Tile {
        let attention = Attention {
            heads: PhantomData::<Trips<Inner>>,
            scale: F32::<EIGHTH>,
        };
        Op::forward(
            &attention,
            device,
            ((query.view(), key.view(), value.view()), visible),
        )
    }
}

/// The tile's rows head-major, as a projection into heads gives them.
pub type Major =
    HbmTensor<bf16, Chip, m![Inner / ({ Symbol<Head> }), Rows, Inner % ({ Symbol<Head> })]>;

/// The byte offsets of the mel frames each of `conv2d1`'s three time taps reads, for a launch's
/// output rows, one an output time: tap `kt` of time `y` of chunk `c` reads frame `2y + kt - 1` of
/// the chunk, or the zero frame `zero` outside it and past the launch's rows.
pub fn frames(rows: usize, zero: usize, bytes: usize) -> [Vec<i32>; 3] {
    std::array::from_fn(|tap| {
        (0..rows)
            .map(|row| {
                let (chunk, time) = (row / TIMES[0], row % TIMES[0]);
                let at = match (2 * time + tap).checked_sub(1) {
                    Some(frame) if chunk < CHUNKS && frame < FRAMES => chunk * FRAMES + frame,
                    _ => zero,
                };
                (at * bytes) as i32
            })
            .collect()
    })
}

/// The byte offsets of the rows each tap of a convolution reads, for a launch's output rows:
/// tap `t = 3 kf + kt` of output (time `y`, frequency `x`) of chunk `c` reads input (time
/// `2y + kt - 1`, frequency `2x + kf - 1`), or the zero row `zero` outside the input; output rows
/// past the launch's read the zero row. The rows are `(chunk, time, frequency)`-major, each
/// `bytes` long.
pub fn offsets(stage: usize, rows: usize, zero: usize, bytes: usize) -> [Vec<i32>; 9] {
    let (inputs, outputs) = (
        (TIMES[stage - 1], FREQUENCIES[stage - 1]),
        (TIMES[stage], FREQUENCIES[stage]),
    );
    std::array::from_fn(|tap| {
        let (kf, kt) = (tap / 3, tap % 3);
        (0..rows)
            .map(|row| {
                let chunk = row / (outputs.0 * outputs.1);
                let (time, frequency) = (row / outputs.1 % outputs.0, row % outputs.1);
                let input = (2 * time + kt)
                    .checked_sub(1)
                    .zip((2 * frequency + kf).checked_sub(1));
                let at = match input {
                    Some((t, f)) if chunk < CHUNKS && t < inputs.0 && f < inputs.1 => {
                        (chunk * inputs.0 + t) * inputs.1 + f
                    }
                    _ => zero,
                };
                (at * bytes) as i32
            })
            .collect()
    })
}

/// `N` projections of `In` into `Out`, one a kernel tap: tap `t`'s coefficient of output `out` and
/// input `in` is `weight(t, out, in)`, zero where it gives none; the first tap's bias is `bias`.
fn taps<const N: usize, In: AxisName, Out: AxisName>(
    cx: &mut Context<'_>,
    weight: impl Fn(usize, usize, usize) -> Option<f32>,
    bias: impl Fn(usize) -> Option<f32>,
) -> Result<[Biased<In, Out>; N], LoadError> {
    let mut taps = Vec::with_capacity(N);
    for tap in 0..N {
        let coefficients: Vec<bf16> = (0..Out::SIZE * In::SIZE)
            .map(|at| bf16::from_f32(weight(tap, at / In::SIZE, at % In::SIZE).unwrap_or(0.0)))
            .collect();
        let offsets: Vec<f32> = (0..Out::SIZE)
            .map(|out| match tap {
                0 => bias(out).unwrap_or(0.0),
                _ => 0.0,
            })
            .collect();
        let name = tap.to_string();
        let projection = cx.field(&name, "", |cx| {
            let weight =
                HbmTensor::load(cx, HostTensor::<bf16, m![Out, In]>::from_vec(coefficients))?;
            let bias = HbmTensor::load(cx, HostTensor::<f32, m![Out]>::from_vec(offsets))?;
            Ok::<_, LoadError>(Affine::new(weight, bias))
        })?;
        taps.push(projection);
    }
    Ok(taps
        .try_into()
        .unwrap_or_else(|_| unreachable!("one projection a tap")))
}
