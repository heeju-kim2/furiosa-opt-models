//! Qwen3-TTS's speaker encoder on the device, the ECAPA-TDNN under `speaker_encoder`: a reference
//! clip's mel frames, 128 bins each, into the 2048-feature speaker embedding the talker's prompt
//! reads, as projections of windows of frame-major rows.
//!
//! - A signal is HBM rows, one a frame's channels. A launch holds a [`Span`]'s rows, a clip's live
//!   frames first; the rows past them are computed and read by no live row, and every statistic
//!   over time zeroes them first.
//! - A convolution of `K` taps `D` frames apart is `K` projections of windows of one framed
//!   buffer, [`LEAD`] rows before and after the signal; the rows around the live ones reflect
//!   them, as torch's `padding_mode="reflect"` pads.
//! - Res2Net's eight parts of 64 channels are projections of their own, widened to 128 channels
//!   with zeros; the projection after them sums each part's.
//! - The squeeze-excitation's, the attention's statistics' and the embedding's projections of a
//!   vector over time take it as the first row of a block of rows.

use furiosa_opt_std::prelude::*;

use super::codec::{Biased, floats, taps, zeros};
use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::tensor::axis;
use crate::tensor::parallel::Chip;

axes![
    Mel = 128,
    Channels = 512,
    Part = 128,
    PartOut = 128,
    Squeezed = 128,
    Aggregate = 1536,
    Attention = 128,
    Embedding = 2048
];

/// The frames a launch holds: 21.8 s of a 24 kHz clip at 256 samples a frame.
pub const FRAMES: usize = 2048;

/// Rows before and after a framed signal: at least the widest padding, 4 frames.
pub const LEAD: usize = 8;

/// The first convolution's taps, one frame apart.
pub const TAPS: usize = 5;

/// Res2Net's parts, and the channels the checkpoint stores of each.
pub const PARTS: usize = 8;
pub const WIDTH: usize = 64;

/// Each block's Res2Net dilation, of three taps.
pub const DILATIONS: [usize; 3] = [2, 3, 4];

/// The longest launch's rows.
pub type Frames = axis::Rows<FRAMES>;

/// A launch's span of mel frames, its rows: 5.5 s, 10.9 s or 21.8 s, the shortest that holds the
/// clip. The framed buffers are the longest span's, a shorter one's rows their first.
pub trait Span: Send + Sync + 'static {
    type Frames: AxisName + Send + Sync;
}

/// A launch of 512 frames, of 1024 and of 2048.
pub struct Tiny;
pub struct Short;
pub struct Long;

impl Span for Tiny {
    type Frames = axis::Rows<512>;
}

impl Span for Short {
    type Frames = axis::Rows<1024>;
}

impl Span for Long {
    type Frames = Frames;
}

/// A framed buffer's rows.
pub type FramedRows = axis::Rows<{ FRAMES + 2 * LEAD }>;

/// One row of a vector over time.
pub type Row = axis::Rows<1>;

/// The ECAPA-TDNN: the first convolution, three squeeze-excitation Res2Net blocks, the projection
/// of their outputs, attentive statistics pooling and the embedding's projection.
#[derive(DeviceSend)]
pub struct SpeakerEncoder {
    pub first: [Biased<Mel, Channels>; TAPS],
    pub blocks: [Block; 3],
    /// The multi-layer aggregation's projection of each block's output, in block order.
    pub aggregate: [Biased<Channels, Aggregate>; 3],
    pub pooling: Pooling,
    /// The embedding's projection of the attentive mean and standard deviation.
    pub embedding: [Biased<Aggregate, Embedding>; 2],
    /// Minus ones, by which a subtraction multiplies.
    pub negative: HbmTensor<f32, Chip, m![Row, Aggregate]>,
    /// Zero-initialized framed buffers of the mel frames and of a Res2Net part.
    pub framed_mel: HbmBuffer,
    pub framed_part: HbmBuffer,
    /// [`FRAMES`] zero FP32 rows of the widest signal, which clear a signal's dead rows.
    pub zeros: HbmBuffer,
}

/// A squeeze-excitation Res2Net block.
#[derive(DeviceSend)]
pub struct Block {
    /// The first projection into each part: its 64 channels, then zeros.
    pub parts: [Biased<Channels, Part>; PARTS],
    /// Each later part's convolution, three taps the block's dilation apart.
    pub res2net: [[Biased<Part, PartOut>; 3]; PARTS - 1],
    /// The projection after the parts: each part's projection, summed.
    pub after: [Biased<Part, Channels>; PARTS],
    pub squeeze: Biased<Channels, Squeezed>,
    pub excite: Biased<Squeezed, Channels>,
}

/// Attentive statistics pooling.
#[derive(DeviceSend)]
pub struct Pooling {
    /// The attention's first projection of each frame's channels.
    pub frames: Biased<Aggregate, Attention>,
    /// Its projection of the channels' mean and standard deviation over time, without bias.
    pub statistics: [Biased<Aggregate, Attention>; 2],
    /// Each channel's attention score of a frame.
    pub scores: Biased<Attention, Aggregate>,
}

/// The encoder under `speaker_encoder` of a whole Qwen3-TTS checkpoint; the talker is not read.
impl<'a> Load<&'a Checkpoint> for SpeakerEncoder {
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.ignore("talker");
            cx.field("", "speaker_encoder", |cx| Self::read(cx, from))
        })
    }
}

impl SpeakerEncoder {
    fn read(cx: &mut Context<'_>, from: &Checkpoint) -> Result<Self, LoadError> {
        let (mel, channels, aggregate) = (Mel::SIZE, Channels::SIZE, Aggregate::SIZE);
        let weight = floats(cx, from, "blocks.0.conv.weight", &[channels, mel, TAPS])?;
        let bias = floats(cx, from, "blocks.0.conv.bias", &[channels])?;
        let first = taps::<TAPS, Mel, Channels>(
            cx,
            |tap, out, input| Some(weight[(out * mel + input) * TAPS + tap]),
            |out| Some(bias[out]),
        )?;
        let blocks = [
            Block::read(cx, from, 1)?,
            Block::read(cx, from, 2)?,
            Block::read(cx, from, 3)?,
        ];
        let weight = floats(cx, from, "mfa.conv.weight", &[aggregate, 3 * channels, 1])?;
        let bias = floats(cx, from, "mfa.conv.bias", &[aggregate])?;
        let projected = taps::<3, Channels, Aggregate>(
            cx,
            |block, out, input| Some(weight[out * 3 * channels + block * channels + input]),
            |out| Some(bias[out]),
        )?;
        let pooling = Pooling::read(cx, from)?;
        let embedding = Embedding::SIZE;
        let weight = floats(cx, from, "fc.weight", &[embedding, 2 * aggregate, 1])?;
        let bias = floats(cx, from, "fc.bias", &[embedding])?;
        let embedding = taps::<2, Aggregate, Embedding>(
            cx,
            |half, out, input| Some(weight[out * 2 * aggregate + half * aggregate + input]),
            |out| Some(bias[out]),
        )?;
        let negative = HbmTensor::load(
            cx,
            HostTensor::<f32, m![Row, Aggregate]>::from_vec(vec![-1.0; aggregate]),
        )?;
        let dead: HbmTensor<f32, Chip, m![Frames, Aggregate]> = HbmTensor::load(
            cx,
            HostTensor::<f32, m![Frames, Aggregate]>::from_vec(vec![0.0; FRAMES * aggregate]),
        )?;
        Ok(Self {
            first,
            blocks,
            aggregate: projected,
            pooling,
            embedding,
            negative,
            framed_mel: zeros::<FramedRows, Mel>(cx)?,
            framed_part: zeros::<FramedRows, Part>(cx)?,
            zeros: dead.into_buffer(),
        })
    }
}

impl Block {
    fn read(cx: &mut Context<'_>, from: &Checkpoint, at: usize) -> Result<Self, LoadError> {
        let channels = Channels::SIZE;
        let at = format!("blocks.{at}");
        let weight = floats(
            cx,
            from,
            &format!("{at}.tdnn1.conv.weight"),
            &[channels, channels, 1],
        )?;
        let bias = floats(cx, from, &format!("{at}.tdnn1.conv.bias"), &[channels])?;
        let mut parts = Vec::with_capacity(PARTS);
        for part in 0..PARTS {
            let [projection] = taps::<1, Channels, Part>(
                cx,
                |_, out, input| {
                    (out < WIDTH).then(|| weight[(part * WIDTH + out) * channels + input])
                },
                |out| (out < WIDTH).then(|| bias[part * WIDTH + out]),
            )?;
            parts.push(projection);
        }
        let mut res2net = Vec::with_capacity(PARTS - 1);
        for part in 0..PARTS - 1 {
            let path = format!("{at}.res2net_block.blocks.{part}.conv");
            let weight = floats(cx, from, &format!("{path}.weight"), &[WIDTH, WIDTH, 3])?;
            let bias = floats(cx, from, &format!("{path}.bias"), &[WIDTH])?;
            res2net.push(taps::<3, Part, PartOut>(
                cx,
                |tap, out, input| {
                    (out < WIDTH && input < WIDTH).then(|| weight[(out * WIDTH + input) * 3 + tap])
                },
                |out| (out < WIDTH).then(|| bias[out]),
            )?);
        }
        let weight = floats(
            cx,
            from,
            &format!("{at}.tdnn2.conv.weight"),
            &[channels, channels, 1],
        )?;
        let bias = floats(cx, from, &format!("{at}.tdnn2.conv.bias"), &[channels])?;
        let after = taps::<PARTS, Part, Channels>(
            cx,
            |part, out, input| {
                (input < WIDTH).then(|| weight[out * channels + part * WIDTH + input])
            },
            |out| Some(bias[out]),
        )?;
        let squeezed = Squeezed::SIZE;
        let weight = floats(
            cx,
            from,
            &format!("{at}.se_block.conv1.weight"),
            &[squeezed, channels, 1],
        )?;
        let bias = floats(cx, from, &format!("{at}.se_block.conv1.bias"), &[squeezed])?;
        let [squeeze] = taps::<1, Channels, Squeezed>(
            cx,
            |_, out, input| Some(weight[out * channels + input]),
            |out| Some(bias[out]),
        )?;
        let weight = floats(
            cx,
            from,
            &format!("{at}.se_block.conv2.weight"),
            &[channels, squeezed, 1],
        )?;
        let bias = floats(cx, from, &format!("{at}.se_block.conv2.bias"), &[channels])?;
        let [excite] = taps::<1, Squeezed, Channels>(
            cx,
            |_, out, input| Some(weight[out * squeezed + input]),
            |out| Some(bias[out]),
        )?;
        let array = |what: &str| LoadError::InvalidData {
            name: at.clone(),
            reason: format!("{what} of every part"),
        };
        Ok(Self {
            parts: parts.try_into().map_err(|_| array("a projection"))?,
            res2net: res2net.try_into().map_err(|_| array("a convolution"))?,
            after,
            squeeze,
            excite,
        })
    }
}

impl Pooling {
    fn read(cx: &mut Context<'_>, from: &Checkpoint) -> Result<Self, LoadError> {
        let (aggregate, attention) = (Aggregate::SIZE, Attention::SIZE);
        let weight = floats(
            cx,
            from,
            "asp.tdnn.conv.weight",
            &[attention, 3 * aggregate, 1],
        )?;
        let bias = floats(cx, from, "asp.tdnn.conv.bias", &[attention])?;
        // The attention reads each frame's channels, then their mean and standard deviation.
        let [frames] = taps::<1, Aggregate, Attention>(
            cx,
            |_, out, input| Some(weight[out * 3 * aggregate + input]),
            |out| Some(bias[out]),
        )?;
        let statistics = taps::<2, Aggregate, Attention>(
            cx,
            |at, out, input| Some(weight[out * 3 * aggregate + (at + 1) * aggregate + input]),
            |_| None,
        )?;
        let weight = floats(cx, from, "asp.conv.weight", &[aggregate, attention, 1])?;
        let bias = floats(cx, from, "asp.conv.bias", &[aggregate])?;
        let [scores] = taps::<1, Attention, Aggregate>(
            cx,
            |_, out, input| Some(weight[out * attention + input]),
            |out| Some(bias[out]),
        )?;
        Ok(Self {
            frames,
            statistics,
            scores,
        })
    }
}
