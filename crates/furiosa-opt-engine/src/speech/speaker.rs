//! Qwen3-TTS's speaker encoder on the device: a reference clip's mel frames through the
//! ECAPA-TDNN into its speaker embedding, each convolution a sum of tap projections of windows of a
//! framed buffer whose rows around the live frames reflect them.
//!
//! - The host computes the clip's mel frames and plans where each live row and its reflections are
//!   copied; the device computes every layer.
//! - Statistics over time are FP32 sums of the live rows, the dead ones zeroed, by halves of the
//!   launch's rows; a vector over time reaches every row by doubling copies.

use super::qwen3_tts::{Signal, add, convolve_windows, project, recast};
use crate::{AsyncModule, Function, forward};
use furiosa_opt_models::models::decoder::attention::qwen3_tts::codec::{Biased, Convolution};
use furiosa_opt_models::models::decoder::attention::qwen3_tts::speaker::{
    Aggregate, Attention, Block, Channels, DILATIONS, Embedding, FRAMES, LEAD, Long, Mel, PARTS,
    Part, PartOut, Row, Short, Span, SpeakerEncoder, Squeezed, TAPS, Tiny,
};
use furiosa_opt_models::ops::index::copy;
use furiosa_opt_models::ops::linear::{Affine, Unrounded};
use furiosa_opt_models::ops::pointwise::activation::{Exp, Relu, Sigmoid, Sqrt, Tanh};
use furiosa_opt_models::ops::pointwise::add::Add;
use furiosa_opt_models::ops::pointwise::cast::Cast;
use furiosa_opt_models::ops::pointwise::multiply::{Divide, Multiply};
use furiosa_opt_models::tensor::axis;
use furiosa_opt_models::tensor::memory::alias;
use furiosa_opt_models::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;
use std::io;
use std::marker::PhantomData;

/// The rows a vector over time projects in, as their first: the fewest a projection into 128
/// channels tiles.
type Vectors = axis::Rows<256>;

/// FP32 rows of `W` channels.
type Wide<R, W> = HbmTensor<f32, Chip, m![R, W]>;

/// A launch's rows of `W` channels of `T`, the span `S`'s frames.
type Launch<S, T, W> = HbmTensor<T, Chip, m![{ Symbol<<S as Span>::Frames> }, W]>;

/// The fewest live frames: torch's reflection pads fewer rows than the signal holds.
const FEWEST: usize = 5;

/// A clip's encoder stages on the device: each block's output, the aggregation's, the attentive
/// mean and standard deviation in FP32, and the embedding, one BF16 row of the talker's width.
pub struct Encoded {
    pub live: usize,
    pub blocks: [HbmBuffer; 4],
    pub aggregate: HbmBuffer,
    pub pooled: [HbmBuffer; 2],
    pub embedding: HbmBuffer,
}

/// The speaker embedding of a clip's `mels`, 128 bins a frame, as one BF16 row of 2048 on the
/// device.
pub async fn embed(
    device: &mut Device,
    encoder: &SpeakerEncoder,
    mels: &[f32],
) -> io::Result<HbmBuffer> {
    Ok(encode(device, encoder, mels).await?.embedding)
}

/// The encoder's stages of a clip's `mels`, 128 bins a frame in frame order, run in BF16 as
/// Transformers runs the encoder, its statistics in FP32.
pub async fn encode(
    device: &mut Device,
    encoder: &SpeakerEncoder,
    mels: &[f32],
) -> io::Result<Encoded> {
    let invalid = |message: &str| io::Error::new(io::ErrorKind::InvalidInput, message.to_string());
    let bins = Mel::SIZE;
    if !mels.len().is_multiple_of(bins) {
        return Err(invalid("a mel frame holds 128 bins"));
    }
    let live = mels.len() / bins;
    if !(FEWEST..=FRAMES).contains(&live) {
        return Err(invalid(
            "a reference clip holds 5 to 2048 mel frames, up to 21.8 s",
        ));
    }
    if live <= <Tiny as Span>::Frames::SIZE {
        spanned::<Tiny>(device, encoder, mels, live).await
    } else if live <= <Short as Span>::Frames::SIZE {
        spanned::<Short>(device, encoder, mels, live).await
    } else {
        spanned::<Long>(device, encoder, mels, live).await
    }
}

/// The encoder's stages of the `live` frames of `mels` in a launch of the span `S`, which holds
/// them.
async fn spanned<S: Span>(
    device: &mut Device,
    encoder: &SpeakerEncoder,
    mels: &[f32],
    live: usize,
) -> io::Result<Encoded> {
    let bytes: Vec<u8> = (mels.iter())
        .flat_map(|&value| bf16::from_f32(value).to_bits().to_le_bytes())
        .collect();
    let mels = upload(device, &bytes).await?;
    let counts = (
        upload_floats(device, &vec![live as f32; Channels::SIZE]).await?,
        upload_floats(device, &vec![live as f32; Aggregate::SIZE]).await?,
    );
    let counts: (Wide<Row, Channels>, Wide<Row, Aggregate>) =
        (alias(&counts.0, ..)?, alias(&counts.1, ..)?);

    // The first convolution, five taps one frame apart.
    reflect::<Mel>(device, &encoder.framed_mel, &mels, live, 2)?;
    let first: Signal<S::Frames, Channels> = convolve_windows(
        device,
        &encoder.framed_mel,
        &encoder.first,
        core::array::from_fn::<_, TAPS, _>(|tap| LEAD - 2 + tap),
    )
    .await?;
    let mut x = relu(device, &first).await?.into_buffer();
    let mut outputs = vec![x.alias(..).map_err(io::Error::other)?];
    for (block, &dilation) in encoder.blocks.iter().zip(&DILATIONS) {
        x = squeeze_excitation::<S>(device, encoder, block, dilation, &x, live, &counts.0).await?;
        outputs.push(x.alias(..).map_err(io::Error::other)?);
    }

    // The aggregation of the blocks' outputs.
    let windows: [Signal<S::Frames, Channels>; 3] = [
        alias(&outputs[1], ..)?,
        alias(&outputs[2], ..)?,
        alias(&outputs[3], ..)?,
    ];
    let aggregate = Function::new(Convolution(&encoder.aggregate), forward)
        .forward(device, (), &mut (), &windows)
        .await?;
    let aggregate = relu(device, &aggregate).await?;

    // Attentive statistics pooling: the channels' mean and standard deviation over the live
    // frames, the attention's scores of each frame from the frame and them, and the scores'
    // softmax over the live frames weighing a second mean and standard deviation.
    let wide = widen(device, &aggregate).await?;
    let wide = clear::<S, _>(device, encoder, wide, live)?;
    let squares = multiply(
        device,
        &view::<S, Aggregate>(&wide)?,
        &view::<S, Aggregate>(&wide)?,
    )
    .await?
    .into_buffer();
    let mean = average::<S, Aggregate>(device, &wide, &counts.1).await?;
    let second = average::<S, Aggregate>(device, &squares, &counts.1).await?;
    let spread = deviation(device, encoder, &mean, &second).await?;
    let statistics = [
        vector_row(device, &mean).await?,
        vector_row(device, &spread).await?,
    ];
    let pooling = &encoder.pooling;
    let constants = [
        unrounded(device, &pooling.statistics[0], &statistics[0]).await?,
        unrounded(device, &pooling.statistics[1], &statistics[1]).await?,
    ];
    let constant = add_wide(device, &constants[0], &constants[1]).await?;
    let constant: Wide<S::Frames, Attention> = broadcast::<S, _, _>(
        device,
        &constant.into_buffer(),
        Attention::SIZE * size_of::<f32>(),
    )?;
    let projected = unrounded(device, &pooling.frames, &aggregate).await?;
    let attention = add_wide(device, &projected, &constant).await?;
    let attention = relu_wide(device, &attention).await?;
    let attention = narrow(device, &attention).await?;
    let attention = Function::new(&Tanh, forward)
        .forward(device, (), &mut (), &attention)
        .await?;
    let scores = unrounded(device, &pooling.scores, &attention).await?;
    let weights = Function::new(&Exp, forward)
        .forward(device, (), &mut (), &scores)
        .await?;
    let weights = clear::<S, _>(device, encoder, weights, live)?;
    let mass = total::<S, Aggregate>(device, &weights).await?;
    let weighted = multiply(
        device,
        &view::<S, Aggregate>(&weights)?,
        &view::<S, Aggregate>(&wide)?,
    )
    .await?
    .into_buffer();
    let mean = average::<S, Aggregate>(device, &weighted, &mass).await?;
    let weighted = multiply(
        device,
        &view::<S, Aggregate>(&weights)?,
        &view::<S, Aggregate>(&squares)?,
    )
    .await?
    .into_buffer();
    let second = average::<S, Aggregate>(device, &weighted, &mass).await?;
    let spread = deviation(device, encoder, &mean, &second).await?;

    // The embedding of the attentive mean and standard deviation.
    let rows: [Signal<Vectors, Aggregate>; 2] = [
        vector_row(device, &mean).await?,
        vector_row(device, &spread).await?,
    ];
    let pooled = [mean.into_buffer(), spread.into_buffer()];
    let embedding: Signal<Vectors, Embedding> =
        Function::new(Convolution(&encoder.embedding), forward)
            .forward(device, (), &mut (), &rows)
            .await?;
    let embedding = (embedding.into_buffer())
        .alias(..Embedding::SIZE * size_of::<bf16>())
        .map_err(io::Error::other)?;
    let [b0, b1, b2, b3]: [HbmBuffer; 4] = outputs
        .try_into()
        .map_err(|_| io::Error::other("four blocks' outputs"))?;
    Ok(Encoded {
        live,
        blocks: [b0, b1, b2, b3],
        aggregate: aggregate.into_buffer(),
        pooled,
        embedding,
    })
}

/// `SqueezeExcitationRes2NetBlock` of the signal in `x`, its Res2Net taps `dilation` frames apart:
/// the projection into the parts, each later part's convolution of itself plus the part before's
/// output, the projection after, then the gate of the channels' mean over the live frames, plus
/// `x`.
async fn squeeze_excitation<S: Span>(
    device: &mut Device,
    encoder: &SpeakerEncoder,
    block: &Block,
    dilation: usize,
    x: &HbmBuffer,
    live: usize,
    count: &Wide<Row, Channels>,
) -> io::Result<HbmBuffer> {
    let input: Signal<S::Frames, Channels> = alias(x, ..)?;
    let mut parts = Vec::with_capacity(PARTS);
    for projection in &block.parts {
        let part = project(device, projection, &input).await?;
        parts.push(relu(device, &part).await?.into_buffer());
    }
    let mut outputs: Vec<HbmBuffer> = vec![parts[0].alias(..).map_err(io::Error::other)?];
    for (at, taps) in block.res2net.iter().enumerate() {
        let part: Signal<S::Frames, Part> = alias(&parts[at + 1], ..)?;
        let input = match at {
            0 => part,
            _ => add(device, &part, &alias(&outputs[at], ..)?).await?,
        };
        reflect::<Part>(
            device,
            &encoder.framed_part,
            &input.into_buffer(),
            live,
            dilation,
        )?;
        let output: Signal<S::Frames, PartOut> = convolve_windows(
            device,
            &encoder.framed_part,
            taps,
            core::array::from_fn(|tap| LEAD - dilation + tap * dilation),
        )
        .await?;
        let output: Signal<S::Frames, Part> = recast(relu(device, &output).await?)?;
        outputs.push(output.into_buffer());
    }
    let windows: [Signal<S::Frames, Part>; PARTS] = (outputs.iter())
        .map(|output| alias(output, ..))
        .collect::<io::Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| io::Error::other("an output a part"))?;
    let after = Function::new(Convolution(&block.after), forward)
        .forward(device, (), &mut (), &windows)
        .await?;
    let after = relu(device, &after).await?;
    // The gate: the live frames' mean, squeezed and excited.
    let wide = widen(device, &after).await?;
    let wide = clear::<S, _>(device, encoder, wide, live)?;
    let mean = average::<S, Channels>(device, &wide, count).await?;
    let mean = vector_row(device, &mean).await?;
    let squeezed: Signal<Vectors, Squeezed> = project(device, &block.squeeze, &mean).await?;
    let squeezed = relu(device, &squeezed).await?;
    let excited: Signal<Vectors, Channels> = project(device, &block.excite, &squeezed).await?;
    let gate = Function::new(&Sigmoid, forward)
        .forward(device, (), &mut (), &excited)
        .await?;
    let gate: Signal<S::Frames, Channels> = broadcast::<S, _, _>(
        device,
        &gate.into_buffer(),
        Channels::SIZE * size_of::<bf16>(),
    )?;
    let gated = Function::new(&Multiply, forward)
        .forward(device, (), &mut (), (&after, &gate))
        .await?;
    Ok(add(device, &gated, &input).await?.into_buffer())
}

/// The standard deviation of the mean `mean` and mean square `second`, `sqrt(max(E[x²] - E[x]²,
/// 0))`.
async fn deviation(
    device: &mut Device,
    encoder: &SpeakerEncoder,
    mean: &Wide<Row, Aggregate>,
    second: &Wide<Row, Aggregate>,
) -> io::Result<Wide<Row, Aggregate>> {
    let square = multiply(device, mean, mean).await?;
    let negative = multiply(device, &square, &encoder.negative).await?;
    let variance = add_wide(device, second, &negative).await?;
    let variance = relu_wide(device, &variance).await?;
    Function::new(&Sqrt, forward)
        .forward(device, (), &mut (), &variance)
        .await
}

/// The sum of `x`'s rows over `count`.
async fn average<S: Span, W: AxisName + Send + Sync>(
    device: &mut Device,
    x: &HbmBuffer,
    count: &Wide<Row, W>,
) -> io::Result<Wide<Row, W>> {
    let sum = total::<S, W>(device, x).await?;
    divide(device, &sum, count).await
}

/// `vector` rounded to BF16 as the first row of a block of rows.
async fn vector_row<W: AxisName + Send + Sync>(
    device: &mut Device,
    vector: &Wide<Row, W>,
) -> io::Result<Signal<Vectors, W>> {
    let row = narrow(device, vector).await?;
    first_row(device, row)
}

/// The sum of the rows of `x`, a launch's FP32 rows of `W` channels, by halves.
async fn total<S: Span, W: AxisName + Send + Sync>(
    device: &mut Device,
    x: &HbmBuffer,
) -> io::Result<Wide<Row, W>> {
    let mut sum = x.alias(..).map_err(io::Error::other)?;
    macro_rules! halve {
        ($($half:literal),+) => {$(
            let width = $half * W::SIZE * size_of::<f32>();
            let first: Wide<axis::Rows<$half>, W> = alias(&sum, ..width)?;
            let second: Wide<axis::Rows<$half>, W> = alias(&sum, width..2 * width)?;
            sum = Function::new(&Add, forward)
                .forward(device, (), &mut (), (&first, &second))
                .await?
                .into_buffer();
        )+};
    }
    const { assert!(FRAMES == 2048 && <Short as Span>::Frames::SIZE == 1024) };
    const { assert!(<Tiny as Span>::Frames::SIZE == 512) };
    if S::Frames::SIZE >= 2048 {
        halve!(1024);
    }
    if S::Frames::SIZE >= 1024 {
        halve!(512);
    }
    halve!(256, 128, 64, 32, 16, 8, 4, 2, 1);
    alias(&sum, ..)
}

/// `x`'s storage with its rows from `live` on zeroed.
fn clear<S: Span, W: AxisName>(
    device: &mut Device,
    encoder: &SpeakerEncoder,
    x: Wide<S::Frames, W>,
    live: usize,
) -> io::Result<HbmBuffer> {
    let row = W::SIZE * size_of::<f32>();
    let buffer = x.into_buffer();
    let frames = S::Frames::SIZE;
    if live < frames {
        let other = io::Error::other;
        let zeros = (encoder.zeros.alias(..(frames - live) * row)).map_err(other)?;
        let mut dead = buffer.alias(live * row..frames * row).map_err(other)?;
        copy(device, [(&zeros, &mut dead)])?;
    }
    Ok(buffer)
}

/// A launch's FP32 rows of `W` channels in `buffer`.
fn view<S: Span, W: AxisName>(buffer: &HbmBuffer) -> io::Result<Wide<S::Frames, W>> {
    alias(buffer, ..)
}

/// `source`'s first `live` rows copied into `framed` after its [`LEAD`] rows, and `pad` rows
/// around them reflecting them, as torch's reflection pads.
fn reflect<W: AxisName>(
    device: &mut Device,
    framed: &HbmBuffer,
    source: &HbmBuffer,
    live: usize,
    pad: usize,
) -> io::Result<()> {
    let other = io::Error::other;
    let row = W::SIZE * size_of::<bf16>();
    let rows = |buffer: &HbmBuffer, at: usize, count: usize| {
        (buffer.alias(at * row..(at + count) * row)).map_err(other)
    };
    let mut pairs = vec![(rows(source, 0, live)?, rows(framed, LEAD, live)?)];
    for apart in 1..=pad {
        pairs.push((rows(source, apart, 1)?, rows(framed, LEAD - apart, 1)?));
        pairs.push((
            rows(source, live - 1 - apart, 1)?,
            rows(framed, LEAD + live - 1 + apart, 1)?,
        ));
    }
    copy(device, pairs.iter_mut().map(|(from, to)| (&*from, to)))
}

/// The first row of `row` in every row of a launch, each copy doubling the rows written.
fn broadcast<S: Span, T: Scalar, W: AxisName>(
    device: &mut Device,
    row: &HbmBuffer,
    bytes: usize,
) -> io::Result<Launch<S, T, W>> {
    let other = io::Error::other;
    let frames = S::Frames::SIZE;
    let rows = HbmBuffer::alloc(device, frames * bytes).map_err(other)?;
    let (source, mut first) = (
        row.alias(..bytes).map_err(other)?,
        rows.alias(..bytes).map_err(other)?,
    );
    copy(device, [(&source, &mut first)])?;
    let mut written = 1;
    while written < frames {
        let source = rows.alias(..written * bytes).map_err(other)?;
        let mut target = (rows.alias(written * bytes..2 * written * bytes)).map_err(other)?;
        copy(device, [(&source, &mut target)])?;
        written *= 2;
    }
    alias(&rows, ..)
}

/// A block of rows whose first is `vector`'s row; the others are not read.
fn first_row<W: AxisName>(
    device: &mut Device,
    vector: Signal<Row, W>,
) -> io::Result<Signal<Vectors, W>> {
    let other = io::Error::other;
    let bytes = W::SIZE * size_of::<bf16>();
    let block = HbmBuffer::alloc(device, Vectors::SIZE * bytes).map_err(other)?;
    let mut first = block.alias(..bytes).map_err(other)?;
    copy(device, [(&vector.into_buffer(), &mut first)])?;
    alias(&block, ..)
}

/// `rows` projected by `projection` into FP32, unrounded.
async fn unrounded<R, In, Out>(
    device: &mut Device,
    projection: &Biased<In, Out>,
    rows: &Signal<R, In>,
) -> io::Result<Wide<R, Out>>
where
    R: AxisName + Send + Sync,
    In: AxisName + Send + Sync,
    Out: AxisName + Send + Sync,
{
    let projection = Unrounded::<In, Out, Chip, _, _>(
        Affine::new(&projection.weight, &projection.bias),
        PhantomData,
    );
    Function::new(&projection, forward)
        .forward(device, (), &mut (), rows)
        .await
}

async fn relu<R: AxisName + Send + Sync, W: AxisName + Send + Sync>(
    device: &mut Device,
    x: &Signal<R, W>,
) -> io::Result<Signal<R, W>> {
    Function::new(&Relu, forward)
        .forward(device, (), &mut (), x)
        .await
}

async fn relu_wide<R: AxisName + Send + Sync, W: AxisName + Send + Sync>(
    device: &mut Device,
    x: &Wide<R, W>,
) -> io::Result<Wide<R, W>> {
    Function::new(&Relu, forward)
        .forward(device, (), &mut (), x)
        .await
}

/// `x` widened exactly to FP32.
async fn widen<R: AxisName + Send + Sync, W: AxisName + Send + Sync>(
    device: &mut Device,
    x: &Signal<R, W>,
) -> io::Result<Wide<R, W>> {
    Function::new(&Cast::<f32>::default(), forward)
        .forward(device, (), &mut (), x)
        .await
}

/// `x` rounded once to BF16.
async fn narrow<R: AxisName + Send + Sync, W: AxisName + Send + Sync>(
    device: &mut Device,
    x: &Wide<R, W>,
) -> io::Result<Signal<R, W>> {
    Function::new(&Cast::<bf16>::default(), forward)
        .forward(device, (), &mut (), x)
        .await
}

async fn multiply<R: AxisName + Send + Sync, W: AxisName + Send + Sync>(
    device: &mut Device,
    a: &Wide<R, W>,
    b: &Wide<R, W>,
) -> io::Result<Wide<R, W>> {
    Function::new(&Multiply, forward)
        .forward(device, (), &mut (), (a, b))
        .await
}

async fn divide<R: AxisName + Send + Sync, W: AxisName + Send + Sync>(
    device: &mut Device,
    a: &Wide<R, W>,
    b: &Wide<R, W>,
) -> io::Result<Wide<R, W>> {
    Function::new(&Divide, forward)
        .forward(device, (), &mut (), (a, b))
        .await
}

async fn add_wide<R: AxisName + Send + Sync, W: AxisName + Send + Sync>(
    device: &mut Device,
    a: &Wide<R, W>,
    b: &Wide<R, W>,
) -> io::Result<Wide<R, W>> {
    Function::new(&Add, forward)
        .forward(device, (), &mut (), (a, b))
        .await
}

async fn upload(device: &mut Device, bytes: &[u8]) -> io::Result<HbmBuffer> {
    (HostBufferSlice::from(bytes).to_hbm(&mut device.pdma).await).map_err(io::Error::other)
}

async fn upload_floats(device: &mut Device, values: &[f32]) -> io::Result<HbmBuffer> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    upload(device, &bytes).await
}
