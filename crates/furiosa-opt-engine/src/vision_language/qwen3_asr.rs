//! Qwen3-ASR's audio tower on the device: a launch's chunks of mel frames through the
//! convolutions, each a sum of tap projections of gathered rows, into rows of the tower's width.

use super::{Audio, Encoder, Image};
use crate::decoder::{HbmTensor as DecoderTensor, Rows as DecoderRows};
use crate::registry::{Listed, Model};
use crate::{AsyncModule, Function, forward};
use furiosa_opt_models::load::{LoadError, Part};
use furiosa_opt_models::models::vision_language::qwen3_asr::audio::{
    self, Band, Bidirectional, CHUNKS, Channels, FRAMES, Filters, Flat, Frontend, FusedMajor,
    Intermediate, Launch, Major, Mel, Offsets, Projected as Projected1, Rows, TIMES, Text, Third,
    Tile, Tower, Width,
};
use furiosa_opt_models::ops::attention::full::paged::Tiling;
use furiosa_opt_models::ops::attention::single::Visible;
use furiosa_opt_models::ops::index::copy;
use furiosa_opt_models::ops::index::gather::Gather;
use furiosa_opt_models::ops::pointwise::{activation::Gelu, add::Add};
use furiosa_opt_models::tensor::memory::alias;
use furiosa_opt_models::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;
use std::io;
use std::pin::Pin;

pub use furiosa_opt_models::models::vision_language::qwen3_asr::audio::Times;

/// A launch's mel frames, its chunks of 100 frames, then one zero frame.
pub type Frames<L> = HbmTensor<bf16, Chip, m![{ Symbol<<L as Launch>::Clip> }, Mel]>;

/// Rows of the tower's width, one an output time, chunk-major.
pub type Embedded = HbmTensor<bf16, Chip, m![Times, Width]>;

/// Writes the zero row `zeros` over `buffer`'s bytes from `start`.
fn zero(
    device: &mut Device,
    zeros: &HbmBuffer,
    buffer: &HbmBuffer,
    start: usize,
) -> io::Result<()> {
    let end = start + zeros.size();
    let mut target = buffer.alias(start..end).map_err(io::Error::other)?;
    copy(device, [(zeros, &mut target)])
}

/// The sum of each tap's projection of the rows of `table` its offsets name, through GELU.
async fn convolve<Table, R, In, Out>(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![Table, In]>,
    offsets: &[HbmTensor<i32, Chip, m![R]>],
    taps: &[audio::Biased<In, Out>],
) -> io::Result<HbmTensor<bf16, Chip, m![R, Out]>>
where
    Table: AxisName + Send + Sync,
    R: AxisName + Send + Sync,
    In: AxisName + Send + Sync,
    Out: AxisName + Send + Sync,
{
    let mut sum: Option<HbmTensor<bf16, Chip, m![R, Out]>> = None;
    for (offsets, tap) in offsets.iter().zip(taps) {
        let rows: HbmTensor<bf16, Chip, m![R, In]> = Function::new(&Gather, forward)
            .forward(device, (), &mut (), (table, offsets))
            .await?;
        let projected: HbmTensor<bf16, Chip, m![R, Out]> = Function::new(tap, forward)
            .forward(device, (), &mut (), &rows)
            .await?;
        sum = Some(match sum {
            None => projected,
            Some(sum) => {
                Function::new(&Add, forward)
                    .forward(device, (), &mut (), (&sum, &projected))
                    .await?
            }
        });
    }
    let sum = sum.ok_or_else(|| io::Error::other("a convolution has taps"))?;
    Function::new(&Gelu, forward)
        .forward(device, (), &mut (), &sum)
        .await
}

/// The rows of a launch's `frames` after the convolutions, the flattening projection and the
/// positions: output time `y` of chunk `c` at row `13 c + y`.
pub async fn embed<L: Launch>(
    frontend: &Frontend,
    device: &mut Device,
    frames: &Frames<L>,
    offsets: &Offsets<L>,
) -> io::Result<Embedded> {
    let pixel = Channels::SIZE * size_of::<bf16>();
    let first: HbmTensor<bf16, Chip, m![{ Symbol<L::First> }, Band]> =
        convolve::<L::Clip, L::First, Mel, Band>(device, frames, &offsets.conv1, &frontend.conv1)
            .await?;
    // The last output time's first pixel reads as the zero row of the second convolution's taps.
    let first = first.into_buffer();
    zero(device, &frontend.zero, &first, L::EMPTY * pixel)?;
    let first: HbmTensor<bf16, Chip, m![{ Symbol<L::Pixels> }, Channels]> = alias(&first, ..)?;
    let second: HbmTensor<bf16, Chip, m![{ Symbol<L::Second> }, Filters]> =
        convolve::<L::Pixels, L::Second, Channels, Filters>(
            device,
            &first,
            &offsets.conv2,
            &frontend.conv2,
        )
        .await?;
    let second = second.into_buffer();
    zero(device, &frontend.zero, &second, L::LIVE * pixel)?;
    let second: HbmTensor<bf16, Chip, m![{ Symbol<L::Second> }, Filters]> = alias(&second, ..)?;
    let third: HbmTensor<bf16, Chip, m![Third, Channels]> =
        convolve::<L::Second, Third, Filters, Channels>(
            device,
            &second,
            &offsets.conv3,
            &frontend.conv3,
        )
        .await?;
    let flat: HbmTensor<bf16, Chip, m![Times, Flat]> = alias(&third.into_buffer(), ..)?;
    let projected: Embedded = Function::new(&frontend.conv_out, forward)
        .forward(device, (), &mut (), &flat)
        .await?;
    let positions: Embedded = Function::new(&Gather, forward)
        .forward(device, (), &mut (), (&frontend.positions, &offsets.phases))
        .await?;
    Function::new(&Add, forward)
        .forward(device, (), &mut (), (&projected, &positions))
        .await
}

/// A tile of rows of the tower's width, the layers' rows.
pub type Hidden = HbmTensor<bf16, Chip, m![Rows, Width]>;

/// A tile of rows of the text model's width, the tower's output.
pub type Projected = HbmTensor<bf16, Chip, m![Rows, Text]>;

/// `tensor`'s storage as a tensor of another mapping of its bytes.
fn recast<D: Scalar, A: M, B: M>(
    tensor: HbmTensor<D, Chip, A>,
) -> io::Result<HbmTensor<D, Chip, B>> {
    alias(&tensor.into_buffer(), ..)
}

/// The tower's rows of a launch's `frames` in the text model's width, the first `live` of them a
/// clip's: one attention window of the launch's rows, its rows past them zero.
pub async fn encode<L: Launch>(
    tower: &Tower,
    device: &mut Device,
    frames: &Frames<L>,
    offsets: &Offsets<L>,
    live: usize,
) -> io::Result<Projected> {
    if live > Times::SIZE {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "a clip's rows exceed one attention window",
        ));
    }
    let embedded = embed(&tower.frontend, device, frames, offsets)
        .await?
        .into_buffer();
    let row = Width::SIZE * size_of::<bf16>();
    let mut head = tower
        .tile
        .alias(..Times::SIZE * row)
        .map_err(io::Error::other)?;
    copy(device, [(&embedded, &mut head)])?;
    let mut hidden: Hidden = alias(&tower.tile, ..)?;
    let visible: Vec<u8> = (0..Rows::SIZE)
        .flat_map(|key| {
            bf16::from_f32(if key < live { 1.0 } else { 0.0 })
                .to_bits()
                .to_le_bytes()
        })
        .collect();
    let visible: Visible<Chip> = alias(
        &HostBufferSlice::from(&visible[..])
            .to_hbm(&mut device.pdma)
            .await
            .map_err(io::Error::other)?,
        ..,
    )?;
    for layer in &tower.layers {
        let attention = &layer.self_attn;
        let normed: Hidden = Function::new(&layer.self_attn_layer_norm, forward)
            .forward(device, (), &mut (), &hidden)
            .await?;
        // The query, key and value heads in one launch, each a third of its storage.
        let fused: FusedMajor = Function::new(&attention.qkv, forward)
            .forward(device, (), &mut (), &normed)
            .await?;
        let fused = fused.into_buffer();
        let third = FusedMajor::size() / 3;
        let heads = |at: usize| -> io::Result<Tile> {
            let major: Major = alias(&fused, at * third..(at + 1) * third)?;
            recast(major)
        };
        let (query, key, value) = (heads(0)?, heads(1)?, heads(2)?);
        let attended: Tile = Function::new(&Bidirectional, forward)
            .forward(device, (), &mut (), (&query, &key, &value, &visible))
            .await?;
        let attended: Major = recast(attended)?;
        let projected: Hidden = Function::new(&attention.out_proj, forward)
            .forward(device, (), &mut (), &attended)
            .await?;
        hidden = Function::new(&Add, forward)
            .forward(device, (), &mut (), (&hidden, &projected))
            .await?;
        let normed: Hidden = Function::new(&layer.final_layer_norm, forward)
            .forward(device, (), &mut (), &hidden)
            .await?;
        let up: HbmTensor<bf16, Chip, m![Rows, Intermediate]> = Function::new(&layer.fc1, forward)
            .forward(device, (), &mut (), &normed)
            .await?;
        let up = Function::new(&Gelu, forward)
            .forward(device, (), &mut (), &up)
            .await?;
        let down: Hidden = Function::new(&layer.fc2, forward)
            .forward(device, (), &mut (), &up)
            .await?;
        hidden = Function::new(&Add, forward)
            .forward(device, (), &mut (), (&hidden, &down))
            .await?;
    }
    let normed: Hidden = Function::new(&tower.ln_post, forward)
        .forward(device, (), &mut (), &hidden)
        .await?;
    let first: HbmTensor<bf16, Chip, m![Rows, Projected1]> =
        Function::new(&tower.linear_1, forward)
            .forward(device, (), &mut (), &normed)
            .await?;
    let first = Function::new(&Gelu, forward)
        .forward(device, (), &mut (), &first)
        .await?;
    Function::new(&tower.linear_2, forward)
        .forward(device, (), &mut (), &first)
        .await
}

/// The tower's rows of a clip of `count` frames, `frames`, in a launch `L` of the fewest chunks
/// that hold them: its frames copied into the launch's zero frames.
async fn launch<L: Launch>(
    tower: &Tower,
    device: &mut Device,
    frames: &HbmBuffer,
    count: usize,
    offsets: &Offsets<L>,
    live: usize,
) -> io::Result<Projected> {
    let mel = Mel::SIZE * size_of::<bf16>();
    let buffer = HostBufferSlice::from(&vec![0u8; L::Clip::SIZE * mel][..])
        .to_hbm(&mut device.pdma)
        .await
        .map_err(io::Error::other)?;
    if count > 0 {
        let mut target = buffer.alias(..count * mel).map_err(io::Error::other)?;
        copy(&mut *device, [(frames, &mut target)])?;
    }
    encode(tower, device, &alias(&buffer, ..)?, offsets, live).await
}

/// Rows of a clip after the convolutions: a chunk's 13 for every whole chunk, and the rows three
/// halvings leave of the last chunk's frames, as Transformers counts them.
pub fn rows(frames: usize) -> usize {
    let halved = |count: usize| match count {
        0 => 0,
        count => (count - 1) / 2 + 1,
    };
    let rest = frames % FRAMES;
    frames / FRAMES * TIMES[2] + halved(halved(halved(rest)))
}

/// The tower reads a load of the encoder part alone and serves audio.
impl Listed for Tower {
    fn reads(part: Part) -> Result<(), LoadError> {
        match part {
            Part::Encoder => Ok(()),
            Part::Whole => Err(LoadError::Unsupported {
                name: "encoder".into(),
                reason: "a media encoder loads alone, so choose the encoder".into(),
            }),
        }
    }

    fn served(self, _: Tiling) -> Model {
        Model::Encoder(Box::new(self))
    }
}

/// Each clip in windows of 8 s, 800 mel frames of 128 features, each window alone in a launch, as
/// Transformers attends them apart.
impl Encoder for Tower {
    fn vision<'a>(
        &'a self,
        _: &'a mut Device,
        _: &'a [Image],
    ) -> Pin<Box<dyn Future<Output = io::Result<Vec<DecoderTensor>>> + Send + 'a>> {
        Box::pin(async {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Qwen3-ASR's encoder embeds audio alone",
            ))
        })
    }

    fn audio<'a>(
        &'a self,
        device: &'a mut Device,
        clips: &'a [Audio],
    ) -> Pin<Box<dyn Future<Output = io::Result<Vec<DecoderTensor>>> + Send + 'a>> {
        Box::pin(async move {
            let mut embedded = Vec::with_capacity(clips.len());
            let (mel, text) = (
                Mel::SIZE * size_of::<bf16>(),
                Text::SIZE * size_of::<bf16>(),
            );
            let window = CHUNKS * FRAMES;
            for clip in clips {
                let frames = clip.frames::<m![Mel]>()?;
                let count = frames.len();
                // Transformers attends each window of 8 s apart, its rows a launch's; the clip's
                // rows are each window's live rows in order.
                let total = rows(count);
                let output =
                    HbmBuffer::alloc(device, total.max(1) * text).map_err(io::Error::other)?;
                let mut written = 0;
                for start in (0..count).step_by(window) {
                    let end = count.min(start + window);
                    let part = (frames.buffer().alias(start * mel..end * mel))
                        .map_err(io::Error::other)?;
                    let (count, live) = (end - start, rows(end - start));
                    let offsets = &self.frontend.offsets;
                    let tile = match count.div_ceil(FRAMES) {
                        0..=2 => launch(self, &mut *device, &part, count, &offsets.0, live).await?,
                        3..=4 => launch(self, &mut *device, &part, count, &offsets.1, live).await?,
                        _ => launch(self, &mut *device, &part, count, &offsets.2, live).await?,
                    };
                    let tile = tile.into_buffer();
                    let source = tile.alias(..live * text).map_err(io::Error::other)?;
                    let mut target = (output.alias(written * text..(written + live) * text))
                        .map_err(io::Error::other)?;
                    copy(&mut *device, [(&source, &mut target)])?;
                    written += live;
                }
                let rows = DecoderRows::<bf16, m![Text]>::new(&output, 0..total)?;
                embedded.push(DecoderTensor::try_from(rows)?);
            }
            Ok(embedded)
        })
    }
}
