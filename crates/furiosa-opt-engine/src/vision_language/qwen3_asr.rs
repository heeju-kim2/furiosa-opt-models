//! Qwen3-ASR's audio tower on the device: a launch's chunks of mel frames through the
//! convolutions, each a sum of tap projections of gathered rows, into rows of the tower's width.

use super::{Audio, Encoder, Image};
use crate::decoder::{HbmTensor as DecoderTensor, Rows as DecoderRows};
use crate::registry::{Listed, Model};
use crate::{AsyncModule, Function, forward};
use furiosa_opt_models::load::{LoadError, Part};
use furiosa_opt_models::models::vision_language::qwen3_asr::audio::{
    self, Band, Bidirectional, CHUNKS, Channels, FRAMES, FREQUENCIES, Filters, Flat, Frontend,
    Intermediate, Major, Mel, Projected as Projected1, Rows, TIMES, Text, Tile, Tower, Width,
};
use furiosa_opt_models::ops::attention::full::paged::Tiling;
use furiosa_opt_models::ops::attention::single::Visible;
use furiosa_opt_models::ops::index::copy;
use furiosa_opt_models::ops::index::gather::Gather;
use furiosa_opt_models::ops::pointwise::{activation::Gelu, add::Add};
use furiosa_opt_models::tensor::axis;
use furiosa_opt_models::tensor::memory::alias;
use furiosa_opt_models::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;
use std::io;
use std::pin::Pin;

axes![Clip = 801, Pixels = 32768];

/// A launch's rows after each convolution, the live ones first: 400 output times of 64
/// frequencies, 6400 pixels then the zero row, 1664 pixels, and 104 output times.
type First = axis::Rows<512>;
type Second = axis::Rows<6528>;
type Third = axis::Rows<2048>;
pub type Times = axis::Rows<128>;

/// The launch's mel frames, `CHUNKS` chunks of 100 frames, then one zero frame.
pub type Frames = HbmTensor<bf16, Chip, m![Clip, Mel]>;

/// Rows of the tower's width, one an output time, chunk-major.
pub type Embedded = HbmTensor<bf16, Chip, m![Times, Width]>;

/// `values` in HBM of `device`.
async fn upload(device: &mut Device, values: &[i32]) -> io::Result<HbmBuffer> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    HostBufferSlice::from(&bytes[..])
        .to_hbm(&mut device.pdma)
        .await
        .map_err(io::Error::other)
}

/// Writes zeros over `bytes` bytes of `buffer` from `start`.
async fn zero(
    device: &mut Device,
    buffer: &HbmBuffer,
    start: usize,
    bytes: usize,
) -> io::Result<()> {
    let zeros = HostBufferSlice::from(&vec![0u8; bytes][..])
        .to_hbm(&mut device.pdma)
        .await
        .map_err(io::Error::other)?;
    let mut target = buffer
        .alias(start..start + bytes)
        .map_err(io::Error::other)?;
    copy(device, [(&zeros, &mut target)])
}

/// The sum of each tap's projection of the rows of `table` its offsets name, through GELU.
async fn convolve<Table, R, In, Out>(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![Table, In]>,
    offsets: &[Vec<i32>],
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
        let offsets: HbmTensor<i32, Chip, m![R]> = alias(&upload(device, offsets).await?, ..)?;
        let rows: HbmTensor<bf16, Chip, m![R, In]> = Function::new(&Gather, forward)
            .forward(device, (), &mut (), (table, &offsets))
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
pub async fn embed(
    frontend: &Frontend,
    device: &mut Device,
    frames: &Frames,
) -> io::Result<Embedded> {
    let mel = Mel::SIZE * size_of::<bf16>();
    let pixel = Channels::SIZE * size_of::<bf16>();
    let first: HbmTensor<bf16, Chip, m![First, Band]> = convolve::<Clip, First, Mel, Band>(
        device,
        frames,
        &audio::frames(First::SIZE, CHUNKS * FRAMES, mel),
        &frontend.conv1,
    )
    .await?;
    // The last output time's first pixel reads as the zero row of the second convolution's taps.
    let first = first.into_buffer();
    let empty = (First::SIZE - 1) * FREQUENCIES[0];
    zero(device, &first, empty * pixel, pixel).await?;
    let first: HbmTensor<bf16, Chip, m![Pixels, Channels]> = alias(&first, ..)?;
    let live = CHUNKS * TIMES[1] * FREQUENCIES[1];
    let second: HbmTensor<bf16, Chip, m![Second, Filters]> =
        convolve::<Pixels, Second, Channels, Filters>(
            device,
            &first,
            &audio::offsets(1, Second::SIZE, empty, pixel),
            &frontend.conv2,
        )
        .await?;
    let second = second.into_buffer();
    zero(device, &second, live * pixel, pixel).await?;
    let second: HbmTensor<bf16, Chip, m![Second, Filters]> = alias(&second, ..)?;
    let third: HbmTensor<bf16, Chip, m![Third, Channels]> =
        convolve::<Second, Third, Filters, Channels>(
            device,
            &second,
            &audio::offsets(2, Third::SIZE, live, pixel),
            &frontend.conv3,
        )
        .await?;
    let flat: HbmTensor<bf16, Chip, m![Times, Flat]> = alias(&third.into_buffer(), ..)?;
    let projected: Embedded = Function::new(&frontend.conv_out, forward)
        .forward(device, (), &mut (), &flat)
        .await?;
    let width = Width::SIZE * size_of::<bf16>();
    let phases: Vec<i32> = (0..Times::SIZE)
        .map(|row| ((row % TIMES[2]) * width) as i32)
        .collect();
    let phases: HbmTensor<i32, Chip, m![Times]> = alias(&upload(device, &phases).await?, ..)?;
    let positions: Embedded = Function::new(&Gather, forward)
        .forward(device, (), &mut (), (&frontend.positions, &phases))
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
pub async fn encode(
    tower: &Tower,
    device: &mut Device,
    frames: &Frames,
    live: usize,
) -> io::Result<Projected> {
    if live > Times::SIZE {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "a clip's rows exceed one attention window",
        ));
    }
    let embedded = embed(&tower.frontend, device, frames).await?.into_buffer();
    let row = Width::SIZE * size_of::<bf16>();
    let tile = HostBufferSlice::from(&vec![0u8; Rows::SIZE * row][..])
        .to_hbm(&mut device.pdma)
        .await
        .map_err(io::Error::other)?;
    let mut head = tile.alias(..Times::SIZE * row).map_err(io::Error::other)?;
    copy(device, [(&embedded, &mut head)])?;
    let mut hidden: Hidden = alias(&tile, ..)?;
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
        let heads = async |device: &mut Device, projection| -> io::Result<Tile> {
            let major: Major = Function::new(projection, forward)
                .forward(device, (), &mut (), &normed)
                .await?;
            recast(major)
        };
        let query = heads(device, &attention.q_proj).await?;
        let key = heads(device, &attention.k_proj).await?;
        let value = heads(device, &attention.v_proj).await?;
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

/// Each clip of at most 8 s, 800 mel frames of 128 features, alone in a launch.
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
            let mel = Mel::SIZE * size_of::<bf16>();
            let mut embedded = Vec::with_capacity(clips.len());
            for clip in clips {
                let frames = clip.frames::<m![Mel]>()?;
                let count = frames.len();
                if count > CHUNKS * FRAMES {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        format!(
                            "a clip of {count} frames exceeds a launch's {}",
                            CHUNKS * FRAMES
                        ),
                    ));
                }
                let launch = HostBufferSlice::from(&vec![0u8; Clip::SIZE * mel][..])
                    .to_hbm(&mut device.pdma)
                    .await
                    .map_err(io::Error::other)?;
                if count > 0 {
                    let mut target = launch.alias(..count * mel).map_err(io::Error::other)?;
                    copy(&mut *device, [(frames.buffer(), &mut target)])?;
                }
                let live = rows(count);
                let tile = encode(self, &mut *device, &alias(&launch, ..)?, live).await?;
                let rows = DecoderRows::<bf16, m![Text]>::new(&tile.into_buffer(), 0..live)?;
                embedded.push(DecoderTensor::try_from(rows)?);
            }
            Ok(embedded)
        })
    }
}
