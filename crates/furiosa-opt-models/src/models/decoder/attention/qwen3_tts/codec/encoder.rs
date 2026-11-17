//! Qwen3-TTS's speech encoder on the device, `Qwen3TTSTokenizerV2Encoder` (Transformers' Mimi) of
//! the 12 Hz speech tokenizer: a reference clip's 24 kHz samples into its codec frames, sixteen
//! codes each, which a voice clone's in-context prompt reads.
//!
//! - A launch holds a [`Span`] of samples, 10.24 s or 30.72 s, the shortest that holds the clip,
//!   its live samples first and zeros after; every layer is causal, so the live frames never read
//!   the rest. Each span has its transformer's constants; the framed buffers are the longest
//!   span's, a shorter one's rows their first.
//! - SEANet's signal at each stage is rows of consecutive samples' channels: [`PHASES`] samples a
//!   row, so a row's channels are the samples' channels one after another and the stages' rows
//!   alias one another's storage. Each causal convolution, of the row's own samples and the row
//!   before's, is two projections of the row and the row before; the last convolution, over
//!   frames, three. The rows before the signal are zero, as Mimi's constant padding.
//! - ELU is ReLU's part plus [`ExpBelow`](crate::ops::pointwise::activation::ExpBelow)'s.
//! - The transformer attends causally over the launch's frames, as Transformers' SDPA and eager
//!   attention do without a sliding window, with products of every head at once as the decoder's
//!   transformer does.
//! - The downsampling convolution's two rows before the signal repeat its first, as Mimi's
//!   replicate padding; its four taps over frames are two projections of frame pairs.
//! - Each codebook's code is the greedy pick of the scores `x e - |e|² / 2` of its entries `e`,
//!   the residual's FP32 rows less the entry; the semantic codebook reads the frames through its
//!   own projection, the acoustic ones through theirs.

use furiosa_opt_std::prelude::*;

use super::{Biased, floats, taps, zeros};
use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::ops::linear::Linear;
use crate::ops::normalization::LayerNorm;
use crate::tensor::axis;
use crate::tensor::form::Weight;
use crate::tensor::parallel::Chip;

axes![
    Input = 64,
    Wide1 = 1024,
    Hidden1 = 512,
    Down1 = 512,
    Wide2 = 1280,
    Hidden2 = 640,
    Down2 = 512,
    Wide3 = 1536,
    Hidden3 = 768,
    Down3 = 512,
    Wide4 = 4096,
    Hidden4 = 2048,
    Down4 = 1024,
    Hidden = 512,
    HiddenOut = 512,
    Heads = 512,
    Keyed = 6144,
    KeyedShort = 2048,
    Inner = 2048,
    FramePair = 1024,
    Code = 256,
    Entries = 2048
];

/// The longest launch's samples: [`FRAMES`] frames of 960 samples, 30.72 s at 24 kHz.
pub const SAMPLES: usize = 737280;

/// The longest launch's frames at 25 Hz, and its codec frames at 12.5 Hz.
pub const FRAMES: usize = 768;
pub const CODED: usize = 384;

/// The samples a frame at 25 Hz encodes.
pub const SAMPLES_A_FRAME: usize = 960;

/// A launch's span of samples: each stage's rows, its frames, its codec frames, and its keys,
/// a copy of the frames for every head.
pub trait Span: Send + Sync + 'static {
    type Rows1: AxisName + Send + Sync;
    type Rows2: AxisName + Send + Sync;
    type Rows3: AxisName + Send + Sync;
    type Frames: AxisName + Send + Sync;
    type Coded: AxisName + Send + Sync;
    type Keyed: AxisName + Send + Sync;

    /// The samples the span holds.
    const SAMPLES: usize = <Self::Frames as AxisName>::SIZE * SAMPLES_A_FRAME;
}

/// A launch of 10.24 s, and of 30.72 s.
pub struct Short;
pub struct Long;

impl Span for Short {
    type Rows1 = axis::Rows<15360>;
    type Rows2 = axis::Rows<6144>;
    type Rows3 = axis::Rows<2048>;
    type Frames = axis::Rows<256>;
    type Coded = axis::Rows<128>;
    type Keyed = KeyedShort;
}

impl Span for Long {
    type Rows1 = Rows1;
    type Rows2 = Rows2;
    type Rows3 = Rows3;
    type Frames = Frames;
    type Coded = Coded;
    type Keyed = Keyed;
}

const _: () = assert!(Long::SAMPLES == SAMPLES && Short::SAMPLES * 3 == SAMPLES);

/// Samples each stage's row holds, and each stage's channels and stride.
pub const PHASES: [usize; 4] = [16, 10, 6, 8];
pub const CHANNELS: [usize; 4] = [64, 128, 256, 512];
pub const STRIDES: [usize; 4] = [4, 5, 6, 8];

/// The first convolution's taps, the residual units', and the last convolution's.
pub const FIRST: usize = 7;
pub const UNIT: usize = 3;
pub const LAST: usize = 3;

/// The transformer's heads, their width and its layers.
pub const HEADS: usize = 8;
pub const HEAD: usize = 64;
pub const LAYERS: usize = 8;

/// The codebooks a frame's codes pick from: the semantic one, then the acoustic ones.
pub const GROUPS: usize = 16;

/// Zero rows before each framed signal: the widest causal reach, two rows.
pub const LEAD: usize = 2;

/// Each stage's rows.
pub type Rows1 = axis::Rows<46080>;
pub type Rows2 = axis::Rows<18432>;
pub type Rows3 = axis::Rows<6144>;
pub type Rows4 = axis::Rows<768>;
pub type Frames = axis::Rows<FRAMES>;
pub type Coded = axis::Rows<CODED>;

/// Every codebook's entries, codebook-major.
pub type Stored = axis::Rows<{ GROUPS * 2048 }>;

/// The additive mask of a score of a later frame, whose exponential is zero.
const MASKED: f32 = -30000.0;

/// The transformer's `norm_eps`.
const EPSILON: usize = 1e-5_f32.to_bits() as usize;

/// `nn.LayerNorm` of the transformer's width.
pub type Norm =
    LayerNorm<HbmTensor<bf16, Chip, m![Hidden]>, HbmTensor<f32, Chip, m![1]>, { EPSILON }>;

/// A SEANet stage: its residual unit's two convolutions, then the downsampling convolution, each
/// tap a projection, of the row before and of the row itself.
#[derive(DeviceSend)]
pub struct Stage<W: AxisName, H: AxisName, D: AxisName> {
    pub unit: [Biased<W, H>; 2],
    pub expand: Biased<H, W>,
    pub down: [Biased<W, D>; 2],
}

/// `MimiTransformerLayer`: pre-norm causal attention, then a pre-norm GELU feed-forward, each
/// scaled and added to its input.
#[derive(DeviceSend)]
pub struct Layer {
    pub input_layernorm: Norm,
    /// `q_proj` over 8, then its rows permuted as `rotate_half` reads them.
    pub q: [Linear<Hidden, Heads, Chip>; 2],
    /// `k_proj`, then its rows permuted as `rotate_half` reads them.
    pub k: [Linear<Hidden, Heads, Chip>; 2],
    /// `v_proj`'s rows, which a call projects by its normed rows into the values' transpose.
    pub v: HbmTensor<bf16, Chip, m![Heads, Hidden]>,
    /// Times `self_attn_layer_scale`.
    pub o_proj: Linear<Heads, Hidden, Chip>,
    pub post_attention_layernorm: Norm,
    pub fc1: Linear<Hidden, Inner, Chip>,
    /// Times `mlp_layer_scale`.
    pub fc2: Linear<Inner, Hidden, Chip>,
}

/// A codebook as a greedy head: each entry's coefficients and minus half its squared norm.
pub type Codebook = Biased<Code, Entries>;

/// The speech encoder, with the constants and buffers a launch reads.
#[derive(DeviceSend)]
pub struct Encoder {
    pub first: [Biased<Input, Wide1>; 2],
    pub stage1: Stage<Wide1, Hidden1, Down1>,
    pub stage2: Stage<Wide2, Hidden2, Down2>,
    pub stage3: Stage<Wide3, Hidden3, Down3>,
    pub stage4: Stage<Wide4, Hidden4, Down4>,
    pub last: [Biased<Down4, Hidden>; LAST],
    pub layers: [Layer; LAYERS],
    /// The downsampling convolution's projections of the frame pair before and of its own.
    pub downsample: [Linear<FramePair, HiddenOut, Chip>; 2],
    /// The semantic codebook's input projection, then the acoustic ones'.
    pub projections: [Linear<HiddenOut, Code, Chip>; 2],
    pub codebooks: [Codebook; GROUPS],
    /// Each codebook's entries, the rows a residual subtracts, codebook-major.
    pub entries: HbmTensor<bf16, Chip, m![Stored, Code]>,
    /// The transformer's constants of a short launch, and of a long one.
    pub short: Attending<Short>,
    pub long: Attending<Long>,
    /// Framed buffers of each stage's convolutions' input, then of the last convolution's:
    /// [`LEAD`] zero rows, then its rows, the long launch's.
    pub framed: [HbmBuffer; 5],
}

/// A launch's rows of every head's features, of the keys' rows, and a mask of each frame's keys.
pub type FrameHeads<S> = HbmTensor<bf16, Chip, m![{ Symbol<<S as Span>::Frames> }, Heads]>;
pub type HeadsKeyed<S> = HbmTensor<bf16, Chip, m![Heads, { Symbol<<S as Span>::Keyed> }]>;
pub type FrameKeys<S> =
    HbmTensor<bf16, Chip, m![{ Symbol<<S as Span>::Frames> }, { Symbol<<S as Span>::Keyed> }]>;

/// The transformer's constants of a launch of the span `S`, and its buffers.
#[derive(DeviceSend)]
pub struct Attending<S: Span> {
    /// The cosine and the sine of each feature's angle at its row's frame.
    pub rotation: [FrameHeads<S>; 2],
    /// Each head's mask of a launch's keys: one at the head's features, zero elsewhere, which a
    /// call multiplies its keys by into the head's block of the block-diagonal keys.
    pub keys: [FrameHeads<S>; HEADS],
    /// One where a feature and a key row are of one head, zero elsewhere.
    pub values: HeadsKeyed<S>,
    /// Each head's sum of a row's weights, in every feature of the head.
    pub masses: Linear<S::Keyed, Heads, Chip>,
    /// Zero where a row sees a key of a head, its frame's or an earlier one's, [`MASKED`] else.
    pub causal: FrameKeys<S>,
    /// Every head's masked rotated keys of a launch, one block a head, then a copy for every
    /// head of its normed rows.
    pub repeated: [HbmBuffer; 2],
}

impl<S: Span> Attending<S> {
    /// The span's rotation tables, block-diagonal head masks, causal mask and buffers.
    fn read(cx: &mut Context<'_>) -> Result<Self, LoadError> {
        let frames = <S::Frames as AxisName>::SIZE;
        let keyed_rows = <S::Keyed as AxisName>::SIZE;
        // Feature `(h, d)` of a frame turns at `10000^(-2 (d mod 32) / 64)` radians a frame.
        let angle = |at: usize| {
            let (frame, feature) = (at / Heads::SIZE, at % Heads::SIZE);
            let pair = feature % HEAD % (HEAD / 2);
            frame as f64 * 10000f64.powf(-2.0 * pair as f64 / HEAD as f64)
        };
        let table = |turn: fn(f64) -> f64| {
            let values = (0..frames * Heads::SIZE)
                .map(|at| bf16::from_f32(turn(angle(at)) as f32))
                .collect::<Vec<_>>();
            HostTensor::<bf16, m![{ Symbol<S::Frames> }, Heads]>::from_vec(values)
        };
        let rotation = [
            HbmTensor::load(cx, table(f64::cos))?,
            HbmTensor::load(cx, table(f64::sin))?,
        ];
        let one = |same: bool| bf16::from_f32(if same { 1.0 } else { 0.0 });
        let keyed = keyed_rows / HEADS;
        let mut keys = Vec::with_capacity(HEADS);
        for head in 0..HEADS {
            let mask = (0..frames * Heads::SIZE)
                .map(|at| one(at % Heads::SIZE / HEAD == head))
                .collect::<Vec<_>>();
            keys.push(HbmTensor::load(
                cx,
                HostTensor::<bf16, m![{ Symbol<S::Frames> }, Heads]>::from_vec(mask),
            )?);
        }
        let values = (0..Heads::SIZE * keyed_rows)
            .map(|at| one(at / keyed_rows / HEAD == at % keyed_rows / keyed))
            .collect::<Vec<_>>();
        let values = HbmTensor::load(
            cx,
            HostTensor::<bf16, m![Heads, { Symbol<S::Keyed> }]>::from_vec(values),
        )?;
        let masses =
            linear::<S::Keyed, Heads>(cx, |out, input| f32::from(out / HEAD == input / keyed))?;
        let causal = (0..frames * keyed_rows)
            .map(|at| {
                let (row, key) = (at / keyed_rows, at % keyed_rows % keyed);
                bf16::from_f32(if key <= row { 0.0 } else { MASKED })
            })
            .collect::<Vec<_>>();
        let causal = HbmTensor::load(
            cx,
            HostTensor::<bf16, m![{ Symbol<S::Frames> }, { Symbol<S::Keyed> }]>::from_vec(causal),
        )?;
        Ok(Self {
            rotation,
            keys: keys
                .try_into()
                .unwrap_or_else(|_| unreachable!("a mask a head")),
            values,
            masses,
            causal,
            repeated: [
                zeros::<S::Keyed, Heads>(cx)?,
                zeros::<S::Keyed, Hidden>(cx)?,
            ],
        })
    }
}

/// The encoder under `encoder` of the 12 Hz speech tokenizer's checkpoint; the decoder is not read.
impl<'a> Load<&'a Checkpoint> for Encoder {
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.ignore("decoder");
            cx.field("", "encoder", |cx| Self::read(cx, from))
        })
    }
}

impl Encoder {
    fn read(cx: &mut Context<'_>, from: &Checkpoint) -> Result<Self, LoadError> {
        // The first convolution, one channel into 64, over rows of 16 samples padded to 64 inputs.
        let weight = floats(cx, from, "encoder.layers.0.conv.weight", &[64, 1, FIRST])?;
        let bias = floats(cx, from, "encoder.layers.0.conv.bias", &[64])?;
        let first = taps::<2, Input, Wide1>(
            cx,
            |tap, out, input| {
                (input < PHASES[0])
                    .then(|| row_tap(&weight, (64, 1, FIRST), (PHASES[0], 1), 1 - tap, out, input))
                    .flatten()
            },
            |out| Some(bias[out % 64]),
        )?;
        let stage1 = Stage::read(cx, from, 0, (1, 3))?;
        let stage2 = Stage::read(cx, from, 1, (4, 6))?;
        let stage3 = Stage::read(cx, from, 2, (7, 9))?;
        let stage4 = Stage::read(cx, from, 3, (10, 12))?;
        let (wide, hidden) = (Down4::SIZE, Hidden::SIZE);
        let weight = floats(
            cx,
            from,
            "encoder.layers.14.conv.weight",
            &[hidden, wide, LAST],
        )?;
        let bias = floats(cx, from, "encoder.layers.14.conv.bias", &[hidden])?;
        let last = taps::<LAST, Down4, Hidden>(
            cx,
            |tap, out, input| Some(weight[(out * wide + input) * LAST + tap]),
            |out| Some(bias[out]),
        )?;
        let mut layers = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            layers.push(Layer::read(
                cx,
                from,
                &format!("encoder_transformer.layers.{layer}"),
            )?);
        }
        // The downsampling convolution, four taps two frames apart: a pair of frames is a row of
        // 1024 features, its first frame's then its second's; the codec frame reads the pair
        // before and its own.
        let weight = floats(cx, from, "downsample.conv.weight", &[hidden, hidden, 4])?;
        let downsample = [0, 1].map(|half| {
            linear::<FramePair, HiddenOut>(cx, |out, input| {
                let (frame, channel) = (input / hidden, input % hidden);
                weight[(out * hidden + channel) * 4 + 2 * half + frame]
            })
        });
        let [before, own] = downsample;
        let downsample = [before?, own?];
        let at = "quantizer";
        let project = |cx: &mut Context<'_>, name: &str| -> Result<_, LoadError> {
            let weight = floats(
                cx,
                from,
                &format!("{at}.{name}_residual_vector_quantizer.input_proj.weight"),
                &[Code::SIZE, hidden, 1],
            )?;
            linear::<HiddenOut, Code>(cx, |out, input| weight[out * hidden + input])
        };
        let projections = [project(cx, "semantic")?, project(cx, "acoustic")?];
        for name in ["semantic", "acoustic"] {
            cx.ignore(&format!(
                "{at}.{name}_residual_vector_quantizer.output_proj"
            ));
        }
        let code = Code::SIZE;
        let mut codebooks = Vec::with_capacity(GROUPS);
        let mut entries = Vec::with_capacity(GROUPS * 2048 * code);
        for group in 0..GROUPS {
            let path = match group {
                0 => format!("{at}.semantic_residual_vector_quantizer.layers.0.codebook"),
                _ => format!(
                    "{at}.acoustic_residual_vector_quantizer.layers.{}.codebook",
                    group - 1
                ),
            };
            let usage = floats(cx, from, &format!("{path}.cluster_usage"), &[2048])?;
            let sums = floats(cx, from, &format!("{path}.embed_sum"), &[2048, code])?;
            cx.ignore(&format!("{path}.initialized"));
            let entry: Vec<f32> = (0..2048 * code)
                .map(|at| sums[at] / usage[at / code].max(1e-5))
                .collect();
            let [book] = taps::<1, Code, Entries>(
                cx,
                |_, out, input| Some(entry[out * code + input]),
                |out| {
                    let row = &entry[out * code..(out + 1) * code];
                    Some(-0.5 * row.iter().map(|value| value * value).sum::<f32>())
                },
            )?;
            codebooks.push(book);
            entries.extend(entry.iter().map(|&value| bf16::from_f32(value)));
        }
        // The acoustic codebooks past the sixteenth, which no frame's codes read.
        for layer in GROUPS - 1..31 {
            cx.ignore(&format!(
                "{at}.acoustic_residual_vector_quantizer.layers.{layer}"
            ));
        }
        let entries = HbmTensor::load(cx, HostTensor::<bf16, m![Stored, Code]>::from_vec(entries))?;
        let framed = [
            zeros::<axis::Rows<{ LEAD + 46080 }>, Wide1>(cx)?,
            zeros::<axis::Rows<{ LEAD + 18432 }>, Wide2>(cx)?,
            zeros::<axis::Rows<{ LEAD + 6144 }>, Wide3>(cx)?,
            zeros::<axis::Rows<{ LEAD + 768 }>, Wide4>(cx)?,
            zeros::<axis::Rows<{ LEAD + FRAMES }>, Down4>(cx)?,
        ];
        Ok(Self {
            first,
            stage1,
            stage2,
            stage3,
            stage4,
            last,
            layers: layers
                .try_into()
                .unwrap_or_else(|_| unreachable!("a layer an index")),
            downsample,
            projections,
            codebooks: codebooks
                .try_into()
                .unwrap_or_else(|_| unreachable!("a codebook a group")),
            entries,
            short: Attending::read(cx)?,
            long: Attending::read(cx)?,
            framed,
        })
    }
}

impl<W: AxisName, H: AxisName, D: AxisName> Stage<W, H, D> {
    /// The stage `at`, whose residual unit and downsampling convolution are SEANet's layers
    /// `layers`.
    fn read(
        cx: &mut Context<'_>,
        from: &Checkpoint,
        at: usize,
        (unit, down): (usize, usize),
    ) -> Result<Self, LoadError> {
        let (phases, channels, stride) = (PHASES[at], CHANNELS[at], STRIDES[at]);
        let half = channels / 2;
        let path = format!("encoder.layers.{unit}.block");
        let weight = floats(
            cx,
            from,
            &format!("{path}.1.conv.weight"),
            &[half, channels, UNIT],
        )?;
        let bias = floats(cx, from, &format!("{path}.1.conv.bias"), &[half])?;
        let unit_taps = taps::<2, W, H>(
            cx,
            |tap, out, input| {
                (out < phases * half)
                    .then(|| {
                        row_tap(
                            &weight,
                            (half, channels, UNIT),
                            (phases, 1),
                            1 - tap,
                            out,
                            input,
                        )
                    })
                    .flatten()
            },
            |out| (out < phases * half).then(|| bias[out % half]),
        )?;
        let weight = floats(
            cx,
            from,
            &format!("{path}.3.conv.weight"),
            &[channels, half, 1],
        )?;
        let bias = floats(cx, from, &format!("{path}.3.conv.bias"), &[channels])?;
        let [expand] = taps::<1, H, W>(
            cx,
            |_, out, input| {
                let (sample, channel) = (out / channels, out % channels);
                let (at, from) = (input / half, input % half);
                (input < phases * half && at == sample).then(|| weight[channel * half + from])
            },
            |out| Some(bias[out % channels]),
        )?;
        let weight = floats(
            cx,
            from,
            &format!("encoder.layers.{down}.conv.weight"),
            &[2 * channels, channels, 2 * stride],
        )?;
        let bias = floats(
            cx,
            from,
            &format!("encoder.layers.{down}.conv.bias"),
            &[2 * channels],
        )?;
        let down_taps = taps::<2, W, D>(
            cx,
            |tap, out, input| {
                row_tap(
                    &weight,
                    (2 * channels, channels, 2 * stride),
                    (phases, stride),
                    1 - tap,
                    out,
                    input,
                )
            },
            |out| Some(bias[out % (2 * channels)]),
        )?;
        Ok(Self {
            unit: unit_taps,
            expand,
            down: down_taps,
        })
    }
}

impl Layer {
    fn read(cx: &mut Context<'_>, from: &Checkpoint, at: &str) -> Result<Self, LoadError> {
        let (hidden, heads, inner) = (Hidden::SIZE, Heads::SIZE, Inner::SIZE);
        let input_layernorm = norm(cx, from, &format!("{at}.input_layernorm"))?;
        let mut rotated = |name: &str, scale: f32| -> Result<_, LoadError> {
            let weight = floats(
                cx,
                from,
                &format!("{at}.self_attn.{name}.weight"),
                &[heads, hidden],
            )?;
            let own =
                linear::<Hidden, Heads>(cx, |out, input| scale * weight[out * hidden + input])?;
            // `rotate_half` of feature `(h, d)`: `-x(h, d + 32)` in the first half, `x(h, d - 32)`
            // in the second.
            let turned = linear::<Hidden, Heads>(cx, |out, input| {
                let (head, feature) = (out / HEAD, out % HEAD);
                let half = HEAD / 2;
                let (partner, sign) = match feature < half {
                    true => (feature + half, -1.0),
                    false => (feature - half, 1.0),
                };
                sign * scale * weight[(head * HEAD + partner) * hidden + input]
            })?;
            Ok([own, turned])
        };
        let q = rotated("q_proj", 1.0 / (HEAD as f32).sqrt())?;
        let k = rotated("k_proj", 1.0)?;
        let weight = floats(
            cx,
            from,
            &format!("{at}.self_attn.v_proj.weight"),
            &[heads, hidden],
        )?;
        let v = HbmTensor::load(
            cx,
            HostTensor::<bf16, m![Heads, Hidden]>::from_vec(
                weight
                    .iter()
                    .map(|&value| bf16::from_f32(value))
                    .collect::<Vec<_>>(),
            ),
        )?;
        let scale = floats(
            cx,
            from,
            &format!("{at}.self_attn_layer_scale.scale"),
            &[hidden],
        )?;
        let weight = floats(
            cx,
            from,
            &format!("{at}.self_attn.o_proj.weight"),
            &[hidden, heads],
        )?;
        let o_proj =
            linear::<Heads, Hidden>(cx, |out, input| scale[out] * weight[out * heads + input])?;
        let post_attention_layernorm = norm(cx, from, &format!("{at}.post_attention_layernorm"))?;
        let weight = floats(cx, from, &format!("{at}.mlp.fc1.weight"), &[inner, hidden])?;
        let fc1 = linear::<Hidden, Inner>(cx, |out, input| weight[out * hidden + input])?;
        let scale = floats(cx, from, &format!("{at}.mlp_layer_scale.scale"), &[hidden])?;
        let weight = floats(cx, from, &format!("{at}.mlp.fc2.weight"), &[hidden, inner])?;
        let fc2 =
            linear::<Inner, Hidden>(cx, |out, input| scale[out] * weight[out * inner + input])?;
        Ok(Self {
            input_layernorm,
            q,
            k,
            v,
            o_proj,
            post_attention_layernorm,
            fc1,
            fc2,
        })
    }
}

/// The coefficient of a causal convolution of `taps` taps, `stride` samples apart, over rows of
/// `phases` input samples into rows of `phases / stride` output samples: output `(j, o)` of a row
/// and input `(i, c)` of the row `before` rows earlier, weighed by tap `k` of `weight[o][c][k]`.
/// Output sample `n` reads the inputs `stride (n + 1) - taps + k`, after `taps - stride` zeros.
fn row_tap(
    weight: &[f32],
    (outputs, inputs, taps): (usize, usize, usize),
    (phases, stride): (usize, usize),
    before: usize,
    out: usize,
    input: usize,
) -> Option<f32> {
    let (sample, channel) = (out / outputs, out % outputs);
    let (at, from) = (input / inputs, input % inputs);
    if sample >= phases / stride || at >= phases {
        return None;
    }
    let k = (at + taps) as isize - (phases * before + stride * (sample + 1)) as isize;
    (0..taps as isize)
        .contains(&k)
        .then(|| weight[(channel * inputs + from) * taps + k as usize])
}

/// The layer norm whose `weight` and `bias` are at `path`.
fn norm(cx: &mut Context<'_>, from: &Checkpoint, path: &str) -> Result<Norm, LoadError> {
    let vector = |values: &[f32]| {
        HostTensor::<bf16, m![Hidden]>::from_vec(
            values
                .iter()
                .map(|&value| bf16::from_f32(value))
                .collect::<Vec<_>>(),
        )
    };
    let weight = floats(cx, from, &format!("{path}.weight"), &[Hidden::SIZE])?;
    let bias = floats(cx, from, &format!("{path}.bias"), &[Hidden::SIZE])?;
    Ok(Norm::new(
        HbmTensor::load(cx, vector(&weight))?,
        HbmTensor::load(cx, vector(&bias))?,
        HbmTensor::load(
            cx,
            HostTensor::<f32, m![1]>::from_vec(vec![f32::from_bits(EPSILON as u32)]),
        )?,
    ))
}

/// The projection of `In` into `Out` whose coefficient of output `out` and input `in` is
/// `weight(out, in)`.
fn linear<In: AxisName, Out: AxisName>(
    cx: &mut Context<'_>,
    weight: impl Fn(usize, usize) -> f32,
) -> Result<Linear<In, Out, Chip>, LoadError> {
    let values: Vec<bf16> = (0..Out::SIZE * In::SIZE)
        .map(|at| bf16::from_f32(weight(at / In::SIZE, at % In::SIZE)))
        .collect();
    let tensor = HbmTensor::load(cx, HostTensor::<bf16, m![Out, In]>::from_vec(values))?;
    Ok(Linear::new(Weight::new(tensor)))
}
