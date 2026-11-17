//! Qwen3-TTS's speech decoder on the device, `Qwen3TTSTokenizerV2Decoder` of the 12 Hz speech
//! tokenizer in the checkpoint's `speech_tokenizer` folder: a launch's codec frames through the
//! [`front`] into latent rows, then past the transformer into 24 kHz samples, as projections of
//! windows of time-major rows.
//!
//! - A signal is HBM rows, one a sample's channels. A launch holds [`FRAMES`] codec frames, the
//!   live ones first; its rows past a call's live frames are computed and read by no caller, as
//!   every layer is causal.
//! - A causal convolution of `K` taps `D` samples apart is `K` projections, tap `k` of the rows
//!   `(K - 1 - k) D` before each output: windows of one framed buffer, whose [`LEAD`] rows before
//!   the signal stay zero.
//! - A transposed convolution of stride `S` and `2S` taps is two projections of a row into the `S`
//!   phases of the output channels, of the row itself and of the row before; a row's phases are the
//!   output's consecutive samples, so the sum read as rows of the output channels is the output.
//!   The upsamplers' convolutions of `S` taps take the first projection alone.
//! - Channels widen to a projection's tiling with zeros, 192 to 256 and 96 to 128; SnakeBeta and
//!   every projection keep the added ones zero.
//! - The last convolution writes [`BLOCK`] consecutive samples a row, projecting the block of as
//!   many input rows and the block before, so the waveform comes out contiguous.

use core::f64::consts::PI;
use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use serde::Deserialize;

use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::models::config::{self, Literal};
use crate::models::{Class, Implements};
use crate::ops::linear::Affine;
use crate::ops::normalization::LayerNorm;
use crate::ops::pointwise::add::Add;
use crate::ops::pointwise::snake::Periodic;
use crate::tensor::axis;
use crate::tensor::parallel::Chip;
use crate::{Module, Op};

pub use front::Front;

pub mod encoder;
pub mod front;

axes![
    Latent = 1024,
    Phases = 2048,
    Expanded = 4096,
    First = 1536,
    Second = 768,
    Third = 384,
    Fourth = 256,
    Fifth = 128,
    Phased1 = 6144,
    Phased2 = 1920,
    Third2 = 768,
    Fourth4 = 1024,
    Fifth16 = 2048,
    Window = 16384,
    Samples = 128,
    LatentOut = 1024,
    SecondOut = 768,
    ThirdOut = 384,
    FourthOut = 256,
    FifthOut = 128,
    Wide2 = 1920,
    Wide3 = 2048,
    Wide4 = 1536,
    Wide2Out = 1920,
    Wide3Out = 2048,
    Wide4Out = 1536
];

/// A channel axis and its twin of the same size, which a projection of the channels into
/// themselves writes, as a weight's two axes must differ.
pub trait Twin: AxisName {
    type Out: AxisName;
}

impl Twin for Latent {
    type Out = LatentOut;
}

impl Twin for Second {
    type Out = SecondOut;
}

impl Twin for Third {
    type Out = ThirdOut;
}

impl Twin for Fourth {
    type Out = FourthOut;
}

impl Twin for Fifth {
    type Out = FifthOut;
}

impl Twin for Wide2 {
    type Out = Wide2Out;
}

impl Twin for Wide3 {
    type Out = Wide3Out;
}

impl Twin for Wide4 {
    type Out = Wide4Out;
}

/// A block's residual units' rows: `PHASES` consecutive samples of the block's channels a row,
/// `Rows` of them a launch, the same bytes as the block's rows of a sample each.
pub trait Phased: Twin {
    const PHASES: usize;
    type Rows: AxisName;
}

impl Phased for Second {
    const PHASES: usize = 1;
    type Rows = axis::Rows<4096>;
}

impl Phased for Wide2 {
    const PHASES: usize = 5;
    type Rows = axis::Rows<4096>;
}

impl Phased for Wide3 {
    const PHASES: usize = 8;
    type Rows = axis::Rows<10240>;
}

impl Phased for Wide4 {
    const PHASES: usize = 12;
    type Rows = axis::Rows<20480>;
}

/// Codec frames a launch holds: 10.24 s at 12.5 Hz.
pub const FRAMES: usize = 128;

/// Samples a codec frame decodes into: 24 kHz over 12.5 Hz.
pub const SAMPLES: usize = 1920;

/// Rows before a framed buffer's signal that stay zero: at least the widest reach of a tap, six
/// taps nine samples apart.
pub const LEAD: usize = 64;

/// Samples before the signal in each block's residual units' framed buffer: a multiple of the
/// units' phases, at least the reach of their widest taps.
pub const LEADS: [usize; 4] = [64, 60, 64, 72];

/// Samples a row of the last convolution's output holds, and the rows of its input it projects
/// together.
pub const BLOCK: usize = 128;

/// Taps of every causal convolution but the residual units' second and the projections.
pub const TAPS: usize = 7;

/// The sample distances of a block's residual units' taps.
pub const DILATIONS: [usize; 3] = [1, 3, 9];

/// A launch's rows: its frames, each upsampler's, then each block's.
pub type Frames = axis::Rows<128>;
pub type Twice = axis::Rows<256>;
pub type Latents = axis::Rows<512>;
pub type Rows1 = axis::Rows<4096>;
pub type Rows2 = axis::Rows<20480>;
pub type Rows3 = axis::Rows<81920>;
pub type Rows4 = axis::Rows<245760>;

/// The last convolution's output rows, [`BLOCK`] samples each: a launch's 1920 and 128 more, which
/// read the zero rows past its input and give a projection whole blocks of 256 rows.
pub type Blocks = axis::Rows<2048>;

/// BF16 coefficients over `In` with an FP32 bias of `Out`.
pub type Biased<In, Out> =
    Affine<HbmTensor<bf16, Chip, m![Out, In]>, HbmTensor<f32, Chip, m![Out]>>;

/// SnakeBeta's periodic term over `W` channels, which a caller adds to its input.
pub type Snake<W> = Periodic<W, Chip>;

/// A causal convolution's taps, the first holding the bias.
pub type Taps<In, Out> = [Biased<In, Out>; TAPS];

/// A causal convolution's taps as one launch: each tap's projection of its window of a framed
/// signal, the windows in tap order, summed.
#[derive(DeviceSend)]
pub struct Convolution<'t, In: AxisName, Out: AxisName, const N: usize>(
    pub &'t [Biased<In, Out>; N],
);

impl<X, In: AxisName, Out: AxisName, const N: usize> !Op<X> for Convolution<'_, In, Out, N> {}

impl<'x, R: AxisName, In: AxisName, Out: AxisName, const N: usize, E>
    Module<&'x [HbmTensor<bf16, Chip, m![R, In]>; N], E, &mut ()> for Convolution<'_, In, Out, N>
{
    type Output = HbmTensor<bf16, Chip, m![R, Out]>;

    fn forward(
        &self,
        device: &mut Device,
        _: E,
        _: &mut (),
        windows: &'x [HbmTensor<bf16, Chip, m![R, In]>; N],
    ) -> Self::Output {
        let mut sum = HbmTensor::new();
        #[unroll]
        for tap in 0..N {
            let projected = Op::forward(&self.0[tap], &mut *device, &windows[tap]);
            sum = match tap {
                0 => projected,
                _ => Op::forward(&Add, &mut *device, (&sum, &projected)),
            };
        }
        sum
    }
}

/// SnakeBeta as one launch: the input plus its periodic term.
#[derive(DeviceSend)]
pub struct SnakeBeta<'t, W: AxisName>(pub &'t Snake<W>);

impl<X, W: AxisName> !Op<X> for SnakeBeta<'_, W> {}

impl<'x, R: AxisName, W: AxisName, E> Module<&'x HbmTensor<bf16, Chip, m![R, W]>, E, &mut ()>
    for SnakeBeta<'_, W>
{
    type Output = HbmTensor<bf16, Chip, m![R, W]>;

    fn forward(
        &self,
        device: &mut Device,
        _: E,
        _: &mut (),
        x: &'x HbmTensor<bf16, Chip, m![R, W]>,
    ) -> Self::Output {
        let term = Op::forward(self.0, &mut *device, x);
        Op::forward(&Add, &mut *device, (x, &term))
    }
}

/// `nn.LayerNorm` of the ConvNeXt blocks: epsilon 1e-6.
pub type Norm =
    LayerNorm<HbmTensor<bf16, Chip, m![Latent]>, HbmTensor<f32, Chip, m![1]>, { EPSILON }>;

const EPSILON: usize = 1e-6_f32.to_bits() as usize;

/// `Qwen3TTSTokenizerV2ConvNeXtBlock`: a depthwise causal convolution of seven taps, the norm, the
/// GELU pair four times as wide, scaled by `gamma`, added to the input.
#[derive(DeviceSend)]
pub struct ConvNeXt {
    /// Diagonal taps.
    pub dwconv: Taps<Latent, LatentOut>,
    pub norm: Norm,
    pub pwconv1: Biased<Latent, Expanded>,
    /// `gamma` times the checkpoint's.
    pub pwconv2: Biased<Expanded, Latent>,
}

/// An upsampler of `upsample`: a transposed convolution of stride and taps two into both phases of
/// the latent, then a ConvNeXt block.
#[derive(DeviceSend)]
pub struct Upsample {
    pub transpose: Biased<Latent, Phases>,
    pub block: ConvNeXt,
}

/// `Qwen3TTSTokenizerV2DecoderDecoderResidualUnit` over rows of `C::PHASES` samples: SnakeBeta, a
/// causal convolution, SnakeBeta and a projection, added to the input.
///
/// - The convolution's seven taps `d` samples apart are `N` row taps: tap `m` weighs the row
///   `(N - 1 - m) r` before, `r` the dilation for rows of a sample and 1 for wider rows, its
///   coefficient of output phase `j` and input phase `i` the sample tap `d` apart that lag.
/// - The projection weighs each phase alone.
#[derive(DeviceSend)]
pub struct Unit<C: Phased, const N: usize> {
    pub act1: Snake<C>,
    pub conv1: [Biased<C, C::Out>; N],
    pub act2: Snake<C>,
    pub conv2: Biased<C, C::Out>,
}

/// `Qwen3TTSTokenizerV2DecoderDecoderBlock`: SnakeBeta over `In`, the transposed convolution into
/// `Out` at `P / Out` times the rate, then a residual unit at each of [`DILATIONS`] over the rows
/// of `W`, `A`, `B` and `C` row taps each.
#[derive(DeviceSend)]
pub struct Block<
    In: AxisName,
    Out: AxisName,
    P: AxisName,
    W: Phased,
    const A: usize,
    const B: usize,
    const C: usize,
> {
    pub snake: Snake<In>,
    /// Of a row, then of the row before.
    pub transpose: [Biased<In, P>; 2],
    pub units: (Unit<W, A>, Unit<W, B>, Unit<W, C>),
    /// The channels of a sample of the output.
    pub channels: PhantomData<Out>,
}

/// The decoder past its transformer: `upsample`, then `decoder`'s convolution, blocks, SnakeBeta
/// and last convolution, with the framed buffers a launch's convolutions read.
#[derive(DeviceSend)]
pub struct Waveform {
    pub upsample: [Upsample; 2],
    pub conv: Taps<Latent, First>,
    pub block1: Block<First, Second, Phased1, Second, 7, 7, 7>,
    pub block2: Block<Second, Third, Phased2, Wide2, 3, 5, 12>,
    pub block3: Block<Third2, Fourth, Wide3, Wide3, 2, 4, 8>,
    pub block4: Block<Fourth4, Fifth, Wide4, Wide4, 2, 3, 6>,
    /// Over rows of 16 samples, as many rows as the last block's units hold.
    pub snake: Snake<Fifth16>,
    /// The last convolution of a block of input rows, then of the block before.
    pub output: [Biased<Window, Samples>; 2],
    pub framed: Framed,
}

/// Buffers of [`LEAD`] zero rows, then a signal's rows, by its rows and channels; the last
/// convolution's holds a [`BLOCK`] of zero rows before its input and zero rows past it.
#[derive(DeviceSend)]
pub struct Framed {
    pub twice: HbmBuffer,
    pub latents: HbmBuffer,
    pub first: HbmBuffer,
    pub second: HbmBuffer,
    pub third: HbmBuffer,
    pub fourth: HbmBuffer,
    pub fifth: HbmBuffer,
    pub output: HbmBuffer,
}

/// Channels the checkpoint stores of each block's output, past the decoder's first convolution.
const STORED: [usize; 5] = [1536, 768, 384, 192, 96];

/// Each block's rate: the stride of its transposed convolution.
pub const RATES: [usize; 4] = [8, 5, 4, 3];

/// The decoder under `decoder`, its quantizer, first convolution and transformer, and the encoder
/// ignored.
impl<'a> Load<&'a Checkpoint> for Waveform {
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.ignore("encoder");
            cx.field("", "decoder", |cx| {
                for ignored in ["quantizer", "pre_conv", "pre_transformer"] {
                    cx.ignore(ignored);
                }
                Self::read(cx, from)
            })
        })
    }
}

/// The speech decoder: the front, then the waveform, under `decoder`; the encoder ignored.
#[derive(DeviceSend)]
pub struct Codec {
    pub front: Front,
    pub waveform: Waveform,
}

/// vLLM-Omni's `Qwen3TTSCode2Wav`, served from a whole Qwen3-TTS checkpoint: the decoder of the 12
/// Hz speech tokenizer its `speech_tokenizer` folder holds, read strictly from that folder.
#[derive(DeviceSend)]
pub struct Code2Wav(pub Codec);

/// What a Qwen3-TTS config declares of its speech tokenizer; deserializing checks the values, which
/// nothing reads after.
#[derive(Deserialize)]
#[allow(dead_code)]
struct Code2WavHf {
    model_type: Literal<"qwen3_tts">,
    tokenizer_type: Literal<"qwen3_tts_tokenizer_12hz">,
}

/// The 12 Hz tokenizer of every Qwen3-TTS size; the checkpoint's root is the scope, as the folder
/// lies beside its tensors.
impl Implements for Code2Wav {
    const SIZE: &'static str = "12Hz";
    const CLASS: Class = Class {
        family: "qwen3_tts",
        name: "Code2Wav",
    };

    fn scope(source: &Checkpoint) -> Result<&'static str, LoadError> {
        config::parse::<Code2WavHf>(source.config()).map_err(|error| {
            let (name, reason) = ("config.json".into(), error.to_string());
            match error.kind() {
                std::io::ErrorKind::Unsupported => LoadError::Unsupported { name, reason },
                _ => LoadError::InvalidData { name, reason },
            }
        })?;
        Ok("")
    }
}

/// The decoder in the checkpoint's `speech_tokenizer` folder, strictly; the root's tensors are the
/// talker's and the speaker encoder's.
impl<'a> Load<&'a Checkpoint> for Code2Wav {
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        let folder = from
            .folder("speech_tokenizer")
            .map_err(|error| LoadError::InvalidData {
                name: "speech_tokenizer".into(),
                reason: error.to_string(),
            })?;
        Codec::load(context, &folder).map(Self)
    }
}

impl<'a> Load<&'a Checkpoint> for Codec {
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.ignore("encoder");
            cx.field("", "decoder", |cx| {
                Ok(Self {
                    front: Front::read(cx, from)?,
                    waveform: Waveform::read(cx, from)?,
                })
            })
        })
    }
}

impl Waveform {
    fn read(cx: &mut Context<'_>, from: &Checkpoint) -> Result<Self, LoadError> {
        let latent = Latent::SIZE;
        let upsample = [
            Upsample::read(cx, from, "0")?,
            Upsample::read(cx, from, "1")?,
        ];
        let (weight, bias) = (
            floats(
                cx,
                from,
                "decoder.0.conv.weight",
                &[STORED[0], latent, TAPS],
            )?,
            floats(cx, from, "decoder.0.conv.bias", &[STORED[0]])?,
        );
        let conv = taps::<TAPS, Latent, First>(
            cx,
            |tap, out, input| Some(weight[(out * latent + input) * TAPS + tap]),
            |out| Some(bias[out]),
        )?;
        let block1 = Block::read(cx, from, 1)?;
        let block2 = Block::read(cx, from, 2)?;
        let block3 = Block::read(cx, from, 3)?;
        let block4 = Block::read(cx, from, 4)?;
        let snake = snake(cx, from, "decoder.5", STORED[4], Fifth::SIZE)?;
        let stored = STORED[4];
        let (weight, bias) = (
            floats(cx, from, "decoder.6.conv.weight", &[1, stored, TAPS])?,
            floats(cx, from, "decoder.6.conv.bias", &[1])?,
        );
        // Output sample `j` of a block reads input rows `j - 6..=j`, tap `k` row `j - 6 + k`: row
        // `i` of its block at tap `i + 6 - j`, row `i` of the block before at tap `i - 122 - j`.
        let coefficient = |tap: Option<usize>, input: usize| {
            let channel = input % Fifth::SIZE;
            tap.filter(|&tap| tap < TAPS && channel < stored)
                .map(|tap| weight[channel * TAPS + tap])
        };
        let block = taps::<1, Window, Samples>(
            cx,
            |_, out, input| coefficient((input / Fifth::SIZE + 6).checked_sub(out), input),
            |_| Some(bias[0]),
        )?;
        let before = taps::<1, Window, Samples>(
            cx,
            |_, out, input| {
                let tap = (input / Fifth::SIZE).checked_sub(BLOCK - 6 + out);
                coefficient(tap, input)
            },
            |_| None,
        )?;
        let [block] = block;
        let [before] = before;
        let framed = Framed::read(cx)?;
        Ok(Self {
            upsample,
            conv,
            block1,
            block2,
            block3,
            block4,
            snake,
            output: [block, before],
            framed,
        })
    }
}

impl Upsample {
    /// `upsample.{at}`: the transposed convolution `0`, then the block `1`.
    fn read(cx: &mut Context<'_>, from: &Checkpoint, at: &str) -> Result<Self, LoadError> {
        let latent = Latent::SIZE;
        let at = format!("upsample.{at}");
        let weight = floats(
            cx,
            from,
            &format!("{at}.0.conv.weight"),
            &[latent, latent, 2],
        )?;
        let bias = floats(cx, from, &format!("{at}.0.conv.bias"), &[latent])?;
        // Phase `p` of output channel `o` is column `p * 1024 + o`; the stored weight is
        // `[input, output, tap]`, its tap the phase.
        let [transpose] = taps::<1, Latent, Phases>(
            cx,
            |_, out, input| {
                let (phase, out) = (out / latent, out % latent);
                Some(weight[(input * latent + out) * 2 + phase])
            },
            |out| Some(bias[out % latent]),
        )?;
        let block = ConvNeXt::read(cx, from, &format!("{at}.1"))?;
        Ok(Self { transpose, block })
    }
}

impl ConvNeXt {
    fn read(cx: &mut Context<'_>, from: &Checkpoint, at: &str) -> Result<Self, LoadError> {
        let (latent, expanded) = (Latent::SIZE, Expanded::SIZE);
        let weight = floats(
            cx,
            from,
            &format!("{at}.dwconv.conv.weight"),
            &[latent, 1, TAPS],
        )?;
        let bias = floats(cx, from, &format!("{at}.dwconv.conv.bias"), &[latent])?;
        let dwconv = taps::<TAPS, Latent, LatentOut>(
            cx,
            |tap, out, input| (out == input).then(|| weight[out * TAPS + tap]),
            |out| Some(bias[out]),
        )?;
        let scale = floats(cx, from, &format!("{at}.norm.weight"), &[latent])?;
        let offset = floats(cx, from, &format!("{at}.norm.bias"), &[latent])?;
        let vector = |values: &[f32]| {
            HostTensor::<bf16, m![Latent]>::from_vec(
                values
                    .iter()
                    .map(|&value| bf16::from_f32(value))
                    .collect::<Vec<_>>(),
            )
        };
        let norm = Norm::new(
            HbmTensor::load(cx, vector(&scale))?,
            HbmTensor::load(cx, vector(&offset))?,
            HbmTensor::load(
                cx,
                HostTensor::<f32, m![1]>::from_vec(vec![f32::from_bits(EPSILON as u32)]),
            )?,
        );
        let weight = floats(
            cx,
            from,
            &format!("{at}.pwconv1.weight"),
            &[expanded, latent],
        )?;
        let bias = floats(cx, from, &format!("{at}.pwconv1.bias"), &[expanded])?;
        let [pwconv1] = taps::<1, Latent, Expanded>(
            cx,
            |_, out, input| Some(weight[out * latent + input]),
            |out| Some(bias[out]),
        )?;
        let gamma = floats(cx, from, &format!("{at}.gamma"), &[latent])?;
        let weight = floats(
            cx,
            from,
            &format!("{at}.pwconv2.weight"),
            &[latent, expanded],
        )?;
        let bias = floats(cx, from, &format!("{at}.pwconv2.bias"), &[latent])?;
        let [pwconv2] = taps::<1, Expanded, Latent>(
            cx,
            |_, out, input| Some(gamma[out] * weight[out * expanded + input]),
            |out| Some(gamma[out] * bias[out]),
        )?;
        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
        })
    }
}

impl<
    In: AxisName,
    Out: AxisName,
    P: AxisName,
    W: Phased,
    const A: usize,
    const B: usize,
    const C: usize,
> Block<In, Out, P, W, A, B, C>
{
    /// `decoder.{at}`: SnakeBeta `block.0`, the transposed convolution `block.1` and the units
    /// `block.2..=4`.
    fn read(cx: &mut Context<'_>, from: &Checkpoint, at: usize) -> Result<Self, LoadError> {
        let (inputs, outputs, rate) = (STORED[at - 1], STORED[at], RATES[at - 1]);
        const { assert!(P::SIZE.is_multiple_of(Out::SIZE)) };
        const { assert!(W::SIZE == W::PHASES * Out::SIZE) };
        // The samples a row of the input holds, each its `channels` of `In`, and each rate phases
        // of the output's samples.
        let samples = P::SIZE / (rate * Out::SIZE);
        let channels = In::SIZE / samples;
        assert_eq!(
            (P::SIZE, In::SIZE),
            (samples * rate * Out::SIZE, samples * channels),
            "whole samples a row"
        );
        let at = format!("decoder.{at}.block");
        let snake = snake(cx, from, &format!("{at}.0"), inputs, channels)?;
        let width = 2 * rate;
        let weight = floats(
            cx,
            from,
            &format!("{at}.1.conv.weight"),
            &[inputs, outputs, width],
        )?;
        let bias = floats(cx, from, &format!("{at}.1.conv.bias"), &[outputs])?;
        // Output sample `s rate + p` of a row, channel `o`, column `(s rate + p) Out + o`, weighs
        // the row's input sample `s` by tap `p` and its sample `s - 1` by tap `p + rate`; the row's
        // first output samples weigh the row before's last input sample by tap `p + rate`.
        let tap = |input: usize, out: usize, half: usize, phase: usize| {
            weight[(input * outputs + out) * width + half * rate + phase]
        };
        let split = |out: usize, input: usize| {
            let (sample, o) = (out / Out::SIZE, out % Out::SIZE);
            let (i, c) = (input / channels, input % channels);
            (c < inputs && o < outputs).then_some((sample / rate, sample % rate, o, i, c))
        };
        let [own] = taps::<1, In, P>(
            cx,
            |_, out, input| {
                let (s, phase, o, i, c) = split(out, input)?;
                match s.checked_sub(i)? {
                    0 => Some(tap(c, o, 0, phase)),
                    1 => Some(tap(c, o, 1, phase)),
                    _ => None,
                }
            },
            |out| {
                let out = out % Out::SIZE;
                (out < outputs).then(|| bias[out])
            },
        )?;
        let [before] = taps::<1, In, P>(
            cx,
            |_, out, input| {
                let (s, phase, o, i, c) = split(out, input)?;
                (s == 0 && i + 1 == samples).then(|| tap(c, o, 1, phase))
            },
            |_| None,
        )?;
        let unit = |unit: usize| (format!("{at}.{}", unit + 2), DILATIONS[unit]);
        let ((first, d1), (second, d2), (third, d3)) = (unit(0), unit(1), unit(2));
        let channels = Out::SIZE;
        let units = (
            Unit::<W, A>::read(cx, from, &first, (outputs, channels), d1)?,
            Unit::<W, B>::read(cx, from, &second, (outputs, channels), d2)?,
            Unit::<W, C>::read(cx, from, &third, (outputs, channels), d3)?,
        );
        Ok(Self {
            snake,
            transpose: [own, before],
            units,
            channels: PhantomData,
        })
    }
}

impl<W: Phased, const N: usize> Unit<W, N> {
    /// The unit at `at` of `stored` channels, `channels` a phase as compiled, its taps `dilation`
    /// samples apart.
    fn read(
        cx: &mut Context<'_>,
        from: &Checkpoint,
        at: &str,
        (stored, channels): (usize, usize),
        dilation: usize,
    ) -> Result<Self, LoadError> {
        let phases = W::PHASES;
        assert_eq!(
            N,
            row_taps(phases, dilation),
            "{at}: the row taps its reach spans"
        );
        let apart = row_dilation(phases, dilation);
        let act1 = snake(cx, from, &format!("{at}.act1"), stored, channels)?;
        let weight = floats(
            cx,
            from,
            &format!("{at}.conv1.conv.weight"),
            &[stored, stored, TAPS],
        )?;
        let bias = floats(cx, from, &format!("{at}.conv1.conv.bias"), &[stored])?;
        // Output (j, o) of row r reads input (i, c) of row r - back at the lag j - i + phases back,
        // sample tap 6 - lag / dilation.
        let conv1 = taps::<N, W, W::Out>(
            cx,
            |tap, out, input| {
                let (j, o, i, c) = (
                    out / channels,
                    out % channels,
                    input / channels,
                    input % channels,
                );
                let back = (N - 1 - tap) * apart;
                let lag = (j + phases * back).checked_sub(i)?;
                let k = (lag.is_multiple_of(dilation) && o < stored && c < stored)
                    .then(|| (TAPS - 1).checked_sub(lag / dilation))
                    .flatten()?;
                Some(weight[(o * stored + c) * TAPS + k])
            },
            |out| (out % channels < stored).then(|| bias[out % channels]),
        )?;
        let act2 = snake(cx, from, &format!("{at}.act2"), stored, channels)?;
        let weight = floats(
            cx,
            from,
            &format!("{at}.conv2.conv.weight"),
            &[stored, stored, 1],
        )?;
        let bias = floats(cx, from, &format!("{at}.conv2.conv.bias"), &[stored])?;
        let [conv2] = taps::<1, W, W::Out>(
            cx,
            |_, out, input| {
                let (j, o, i, c) = (
                    out / channels,
                    out % channels,
                    input / channels,
                    input % channels,
                );
                (j == i && o < stored && c < stored).then(|| weight[o * stored + c])
            },
            |out| (out % channels < stored).then(|| bias[out % channels]),
        )?;
        Ok(Self {
            act1,
            conv1,
            act2,
            conv2,
        })
    }
}

impl Framed {
    /// Each buffer zero: [`LEAD`] rows and a signal's, of its channels; the last convolution's
    /// a [`BLOCK`] of rows, then a block of its output's of 128 channels.
    fn read(cx: &mut Context<'_>) -> Result<Self, LoadError> {
        Ok(Self {
            twice: zeros::<axis::Rows<{ LEAD + 256 }>, Latent>(cx)?,
            latents: zeros::<axis::Rows<{ LEAD + 512 }>, Latent>(cx)?,
            first: zeros::<axis::Rows<{ LEAD + 512 }>, First>(cx)?,
            second: zeros::<axis::Rows<{ LEADS[0] + 4096 }>, Second>(cx)?,
            third: zeros::<axis::Rows<{ LEADS[1] + 20480 }>, Third>(cx)?,
            fourth: zeros::<axis::Rows<{ LEADS[2] + 81920 }>, Fourth>(cx)?,
            fifth: zeros::<axis::Rows<{ LEADS[3] + 245760 }>, Fifth>(cx)?,
            output: zeros::<axis::Rows<{ BLOCK + 2048 * BLOCK }>, Fifth>(cx)?,
        })
    }
}

/// Row taps a unit over rows of `phases` samples takes for its seven taps `dilation` apart: seven
/// rows of a sample, else the rows its reach spans and its own.
pub const fn row_taps(phases: usize, dilation: usize) -> usize {
    match phases {
        1 => TAPS,
        _ => ((TAPS - 1) * dilation).div_ceil(phases) + 1,
    }
}

/// The rows a unit's row taps lie apart: its dilation over rows of a sample, else one.
pub const fn row_dilation(phases: usize, dilation: usize) -> usize {
    match phases {
        1 => dilation,
        _ => 1,
    }
}

/// `R` zero rows of `W` BF16 channels.
pub(super) fn zeros<R: AxisName, W: AxisName>(
    cx: &mut Context<'_>,
) -> Result<HbmBuffer, LoadError> {
    let zeros = vec![bf16::from_f32(0.0); R::SIZE * W::SIZE];
    let tensor: HbmTensor<bf16, Chip, m![R, W]> =
        HbmTensor::load(cx, HostTensor::<bf16, m![R, W]>::from_vec(zeros))?;
    Ok(tensor.into_buffer())
}

/// The FP32 values of the tensor at the dotted `path` below the current name.
pub(super) fn floats(
    cx: &mut Context<'_>,
    from: &Checkpoint,
    path: &str,
    shape: &[usize],
) -> Result<Vec<f32>, LoadError> {
    fn below(
        cx: &mut Context<'_>,
        from: &Checkpoint,
        path: &[&str],
        shape: &[usize],
    ) -> Result<Vec<f32>, LoadError> {
        match path {
            [] => cx.floats(from, shape),
            [first, rest @ ..] => cx.field(first, first, |cx| below(cx, from, rest, shape)),
        }
    }
    below(cx, from, &path.split('.').collect::<Vec<_>>(), shape)
}

/// SnakeBeta's periodic term from the log-scale `alpha` and `beta` of `stored` channels at `path`,
/// over rows whose features are phases of `channels` each; the term of a channel past the stored
/// ones is zero.
fn snake<W: AxisName>(
    cx: &mut Context<'_>,
    from: &Checkpoint,
    path: &str,
    stored: usize,
    channels: usize,
) -> Result<Snake<W>, LoadError> {
    let alpha = floats(cx, from, &format!("{path}.alpha"), &[stored])?;
    let beta = floats(cx, from, &format!("{path}.beta"), &[stored])?;
    // Feature `f` of a row of phases is channel `f mod channels` of its sample.
    let scale = |feature: usize| match feature % channels {
        channel if channel < stored => (0.5 / (f64::from(beta[channel]).exp() + 1e-9)) as f32,
        _ => 0.0,
    };
    let frequency = |feature: usize| match feature % channels {
        channel if channel < stored => (f64::from(alpha[channel]).exp() / PI) as f32,
        _ => 0.0,
    };
    let mut vector = |value: &dyn Fn(usize) -> f32| {
        let values: Vec<f32> = (0..W::SIZE).map(value).collect();
        HbmTensor::load(cx, HostTensor::<f32, m![W]>::from_vec(values))
    };
    Ok(Periodic {
        frequency: vector(&frequency)?,
        scale: vector(&scale)?,
        weight: vector(&|channel| -scale(channel))?,
    })
}

/// `N` projections of `In` into `Out`, one a tap: tap `t`'s coefficient of output `out` and input
/// `in` is `weight(t, out, in)`, zero where it gives none; the first tap's bias is `bias`.
pub(super) fn taps<const N: usize, In: AxisName, Out: AxisName>(
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
        let weight = HbmTensor::load(cx, HostTensor::<bf16, m![Out, In]>::from_vec(coefficients))?;
        let bias = HbmTensor::load(cx, HostTensor::<f32, m![Out]>::from_vec(offsets))?;
        taps.push(Affine::new(weight, bias));
    }
    Ok(taps
        .try_into()
        .unwrap_or_else(|_| unreachable!("one projection a tap")))
}
