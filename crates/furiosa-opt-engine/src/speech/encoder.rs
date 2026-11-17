//! Qwen3-TTS's speech encoder on the device: a reference clip's 24 kHz samples through SEANet, the
//! transformer, the downsampling convolution and the residual quantizers into its codec frames,
//! each convolution a sum of tap projections of windows of a framed buffer.
//!
//! - The host lays the samples out as the first convolution's rows, picks each codec frame's
//!   nearest entry of a codebook from the device's scores, the first highest, as greedy decoding
//!   picks a token, and plans the byte offsets of the picked entries; the device computes every
//!   layer, the scores and the residuals.

use super::qwen3_tts::{Signal, add, combine, linear, masked, norm_blocks, window};
use crate::{AsyncModule, Function, forward};
use furiosa_opt_models::models::decoder::attention::qwen3_tts::codec::Biased;
use furiosa_opt_models::models::decoder::attention::qwen3_tts::codec::encoder::{
    Attending, Code, Down1, Down2, Down3, Down4, Encoder, Entries, FRAMES, FramePair, GROUPS,
    Heads, Hidden, HiddenOut, Input, LEAD, Layer, Long, SAMPLES, Short, Span, Stage, Wide1,
};
use furiosa_opt_models::ops::index::copy;
use furiosa_opt_models::ops::index::gather::Gather;
use furiosa_opt_models::ops::linear::{Affine, Linear, Unrounded};
use furiosa_opt_models::ops::pointwise::activation::{Exp, ExpBelow, Gelu, Relu};
use furiosa_opt_models::ops::pointwise::cast::Cast;
use furiosa_opt_models::ops::pointwise::multiply::{Divide, Multiply, Subtract};
use furiosa_opt_models::tensor::form::Weight;
use furiosa_opt_models::tensor::memory::alias;
use furiosa_opt_models::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;
use std::io;
use std::marker::PhantomData;

/// FP32 rows of `W` channels.
type Wide<R, W> = HbmTensor<f32, Chip, m![R, W]>;

/// The samples a codec frame encodes, at 12.5 Hz.
pub const SAMPLES_A_FRAME: usize = 1920;

/// A clip's encoder stages on the device: SEANet's frames, the transformer's, the downsampled
/// codec frames, and the live codec frames' codes.
pub struct Encoded {
    pub live: usize,
    /// SEANet's layers' outputs by Transformers' layer index: the first convolution's, each
    /// residual unit's and each downsampling convolution's, sample-major.
    pub layers: Vec<(usize, HbmBuffer)>,
    pub seanet: HbmBuffer,
    pub transformed: HbmBuffer,
    pub latents: HbmBuffer,
    pub codes: Vec<[i32; GROUPS]>,
}

/// The codec frames of a clip's `samples`, 24 kHz in [-1, 1], up to 30.72 s: sixteen codes each,
/// the semantic codebook's first, `ceil(samples / 1920)` frames as Transformers' encode returns.
pub async fn encode(
    device: &mut Device,
    encoder: &Encoder,
    samples: &[f32],
) -> io::Result<Vec<[i32; GROUPS]>> {
    Ok(stages(device, encoder, samples).await?.codes)
}

/// The encoder's stages of a clip's `samples`, as [`encode`] runs them: in a launch of the
/// shortest span that holds them.
pub async fn stages(
    device: &mut Device,
    encoder: &Encoder,
    samples: &[f32],
) -> io::Result<Encoded> {
    let invalid = |message: &str| io::Error::new(io::ErrorKind::InvalidInput, message.to_string());
    if samples.is_empty() || samples.len() > SAMPLES {
        return Err(invalid(
            "a reference clip holds 1 to 737280 samples, 30.72 s",
        ));
    }
    if samples.len() <= Short::SAMPLES {
        spanned(device, encoder, &encoder.short, samples).await
    } else {
        spanned(device, encoder, &encoder.long, samples).await
    }
}

/// The encoder's stages of `samples` in a launch of the span `S`, which holds them.
async fn spanned<S: Span>(
    device: &mut Device,
    encoder: &Encoder,
    attending: &Attending<S>,
    samples: &[f32],
) -> io::Result<Encoded> {
    let live = samples.len().div_ceil(SAMPLES_A_FRAME);
    // The first convolution's rows: 16 samples a row, zeros to 64 inputs, after the lead rows.
    let (phases, inputs) = (16, Input::SIZE);
    let mut rows = vec![0u8; (LEAD + S::Rows1::SIZE) * inputs * 2];
    for (at, &sample) in samples.iter().enumerate() {
        let (row, phase) = (LEAD + at / phases, at % phases);
        let bits = bf16::from_f32(sample).to_bits().to_le_bytes();
        rows[(row * inputs + phase) * 2..][..2].copy_from_slice(&bits);
    }
    let framed = upload(device, &rows).await?;
    let first: Wide<S::Rows1, Wide1> =
        convolve_wide(device, &framed, &encoder.first, [LEAD - 1, LEAD]).await?;
    let [wide1, wide2, wide3, wide4, frames] = &encoder.framed;
    let (copy0, first) = kept(first)?;
    let mut layers = vec![(0, copy0)];
    let (unit, x): (_, Wide<S::Rows1, Down1>) =
        stage(device, &encoder.stage1, wide1, first).await?;
    let (down, x) = kept(x)?;
    layers.extend([(1, unit), (3, down)]);
    let (unit, x): (_, Wide<S::Rows2, Down2>) =
        stage(device, &encoder.stage2, wide2, rewide(x)?).await?;
    let (down, x) = kept(x)?;
    layers.extend([(4, unit), (6, down)]);
    let (unit, x): (_, Wide<S::Rows3, Down3>) =
        stage(device, &encoder.stage3, wide3, rewide(x)?).await?;
    let (down, x) = kept(x)?;
    layers.extend([(7, unit), (9, down)]);
    let (unit, x): (_, Wide<S::Frames, Down4>) =
        stage(device, &encoder.stage4, wide4, rewide(x)?).await?;
    let (down, x) = kept(x)?;
    layers.extend([(10, unit), (12, down)]);
    activate_framed(device, &x, (frames, LEAD)).await?;
    let x: Wide<S::Frames, Hidden> =
        convolve_wide(device, frames, &encoder.last, [LEAD - 2, LEAD - 1, LEAD]).await?;
    let x: Signal<S::Frames, Hidden> = narrow(device, &x).await?;
    let seanet = x.into_buffer();
    let mut x: Signal<S::Frames, Hidden> = alias(&seanet, ..)?;
    for layer_ in &encoder.layers {
        x = layer(device, attending, layer_, x).await?;
    }
    let transformed = x.into_buffer();
    let latents = downsample::<S>(device, encoder, alias(&transformed, ..)?).await?;
    let codes = quantize::<S>(device, encoder, &latents, live).await?;
    Ok(Encoded {
        live,
        layers,
        seanet,
        transformed,
        latents: latents.into_buffer(),
        codes,
    })
}

/// A SEANet stage of `x`: its residual unit, `x` plus the ELU-led convolutions, then the ELU-led
/// downsampling convolution, framed in `framed`; each convolution's taps summed and the residual
/// added in FP32, each projection's input rounded once to BF16.
async fn stage<R, W, H, D>(
    device: &mut Device,
    stage: &Stage<W, H, D>,
    framed: &HbmBuffer,
    x: Wide<R, W>,
) -> io::Result<(HbmBuffer, Wide<R, D>)>
where
    R: AxisName + Send + Sync,
    W: AxisName + Send + Sync,
    H: AxisName + Send + Sync,
    D: AxisName + Send + Sync,
{
    activate_framed(device, &x, (framed, LEAD)).await?;
    let hidden: Wide<R, H> = convolve_wide(device, framed, &stage.unit, [LEAD - 1, LEAD]).await?;
    let hidden = activate(device, &hidden).await?;
    let expanded = unrounded(device, &stage.expand, &hidden).await?;
    let x = add_wide(device, &x, &expanded).await?;
    activate_framed(device, &x, (framed, LEAD)).await?;
    let (unit, _) = kept(x)?;
    let down = convolve_wide(device, framed, &stage.down, [LEAD - 1, LEAD]).await?;
    Ok((unit, down))
}

/// The sum of each of `taps`' FP32 projection of its window of `framed`, tap `k`'s from row
/// `starts[k]`, in FP32.
async fn convolve_wide<R, In, Out, const N: usize>(
    device: &mut Device,
    framed: &HbmBuffer,
    taps: &[Biased<In, Out>; N],
    starts: [usize; N],
) -> io::Result<Wide<R, Out>>
where
    R: AxisName + Send + Sync,
    In: AxisName + Send + Sync,
    Out: AxisName + Send + Sync,
{
    let mut sum: Option<Wide<R, Out>> = None;
    for (tap, &start) in taps.iter().zip(&starts) {
        let rows: Signal<R, In> = window(framed, start)?;
        let projected = unrounded(device, tap, &rows).await?;
        sum = Some(match sum {
            None => projected,
            Some(sum) => add_wide(device, &sum, &projected).await?,
        });
    }
    sum.ok_or_else(|| io::Error::other("a convolution of taps"))
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

async fn add_wide<R: AxisName + Send + Sync, W: AxisName + Send + Sync>(
    device: &mut Device,
    a: &Wide<R, W>,
    b: &Wide<R, W>,
) -> io::Result<Wide<R, W>> {
    Function::new(&furiosa_opt_models::ops::pointwise::add::Add, forward)
        .forward(device, (), &mut (), (a, b))
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

/// `x`'s FP32 storage as rows of another shape.
fn rewide<A: M, B: M>(x: HbmTensor<f32, Chip, A>) -> io::Result<HbmTensor<f32, Chip, B>> {
    alias(&x.into_buffer(), ..)
}

/// `tensor`, and a second owner of its storage.
fn kept<T: Scalar, E: M>(
    tensor: HbmTensor<T, Chip, E>,
) -> io::Result<(HbmBuffer, HbmTensor<T, Chip, E>)> {
    let buffer = tensor.into_buffer();
    Ok((
        buffer.alias(..).map_err(io::Error::other)?,
        alias(&buffer, ..)?,
    ))
}

/// `ELU(x)` rounded once to BF16, a projection's input.
async fn activate<R: AxisName + Send + Sync, W: AxisName + Send + Sync>(
    device: &mut Device,
    x: &Wide<R, W>,
) -> io::Result<Signal<R, W>> {
    let activated = elu(device, x).await?;
    narrow(device, &activated).await
}

/// `ELU(x)` rounded once to BF16 into `framed`'s rows from row `start` on, the window it fills:
/// the convolution after reads its taps' windows of `framed` with no copy between.
async fn activate_framed<R: AxisName + Send + Sync, W: AxisName + Send + Sync>(
    device: &mut Device,
    x: &Wide<R, W>,
    (framed, start): (&HbmBuffer, usize),
) -> io::Result<Signal<R, W>> {
    let activated = elu(device, x).await?;
    let mut narrowed = window(framed, start)?;
    Function::new(&Cast::<bf16>::default(), forward).forward_into(
        device,
        (),
        &mut (),
        &activated,
        &mut narrowed,
    )?;
    Ok(narrowed)
}

/// `ELU(x)` in FP32: its part above zero plus its part below.
async fn elu<R: AxisName + Send + Sync, W: AxisName + Send + Sync>(
    device: &mut Device,
    x: &Wide<R, W>,
) -> io::Result<Wide<R, W>> {
    let above = Function::new(&Relu, forward)
        .forward(device, (), &mut (), x)
        .await?;
    let below = Function::new(&ExpBelow, forward)
        .forward(device, (), &mut (), x)
        .await?;
    add_wide(device, &above, &below).await
}

/// `x cos + rotate_half(x) sin` of the projections `pair` of `x`.
async fn rotate<S: Span>(
    device: &mut Device,
    attending: &Attending<S>,
    pair: &[Linear<Hidden, Heads, Chip>; 2],
    x: &Signal<S::Frames, Hidden>,
) -> io::Result<Signal<S::Frames, Heads>> {
    let [own, turned] = pair;
    let [cos, sin] = &attending.rotation;
    let own = linear(device, own, x).await?;
    let turned = linear(device, turned, x).await?;
    let own = combine(device, &Multiply, &own, cos).await?;
    let turned = combine(device, &Multiply, &turned, sin).await?;
    add(device, &own, &turned).await
}

/// A transformer layer of `x`: causal attention as block-diagonal products of every head, then
/// the GELU feed-forward.
async fn layer<S: Span>(
    device: &mut Device,
    attending: &Attending<S>,
    layer: &Layer,
    x: Signal<S::Frames, Hidden>,
) -> io::Result<Signal<S::Frames, Hidden>> {
    let [keyed, rows] = &attending.repeated;
    // The normed rows once for every head, each block of `rows` its own launch of the norm.
    let repeated: Signal<S::Keyed, Hidden> = norm_blocks(
        &Function::new(&layer.input_layernorm, forward),
        device,
        &x,
        rows,
    )?;
    let normed: Signal<S::Frames, Hidden> =
        alias(rows, ..<S::Frames as AxisName>::SIZE * Hidden::SIZE * 2)?;
    let queries = rotate(device, attending, &layer.q, &normed).await?;
    let keys = rotate(device, attending, &layer.k, &normed)
        .await?
        .into_buffer();
    let keyed: Signal<S::Keyed, Heads> = masked(device, &attending.keys, &keys, keyed)?;
    let scores: Signal<S::Frames, S::Keyed> =
        linear(device, &Linear::new(Weight::new(keyed)), &queries).await?;
    let scores = add(device, &scores, &attending.causal).await?;
    let weights = Function::new(&Exp, forward)
        .forward(device, (), &mut (), &scores)
        .await?;
    let values: Signal<Heads, S::Keyed> =
        linear(device, &Linear::new(Weight::new(repeated)), &layer.v).await?;
    let values = combine(device, &Multiply, &values, &attending.values).await?;
    let sums: Signal<S::Frames, Heads> =
        linear(device, &Linear::new(Weight::new(values)), &weights).await?;
    let masses = linear(device, &attending.masses, &weights).await?;
    let attended = combine(device, &Divide, &sums, &masses).await?;
    let projected = linear(device, &layer.o_proj, &attended).await?;
    let x = add(device, &x, &projected).await?;
    let normed: Signal<S::Frames, Hidden> = Function::new(&layer.post_attention_layernorm, forward)
        .forward(device, (), &mut (), &x)
        .await?;
    let inner = linear(device, &layer.fc1, &normed).await?;
    let inner = Function::new(&Gelu, forward)
        .forward(device, (), &mut (), &inner)
        .await?;
    let down = linear(device, &layer.fc2, &inner).await?;
    add(device, &x, &down).await
}

/// The downsampling convolution of the frames `x` into codec frames: pairs of frames, the pair
/// before a codec frame's and its own, the pair before the first two copies of the first frame.
async fn downsample<S: Span>(
    device: &mut Device,
    encoder: &Encoder,
    x: Signal<S::Frames, Hidden>,
) -> io::Result<Signal<S::Coded, HiddenOut>> {
    let other = io::Error::other;
    let (frame_bytes, pair) = (Hidden::SIZE * 2, FramePair::SIZE * 2);
    let coded = S::Coded::SIZE;
    let framed = HbmBuffer::alloc(device, (1 + coded) * pair).map_err(other)?;
    let x = x.into_buffer();
    let first = x.alias(..frame_bytes).map_err(other)?;
    let mut pairs = vec![
        (
            x.alias(..).map_err(other)?,
            framed.alias(pair..).map_err(other)?,
        ),
        (
            first.alias(..).map_err(other)?,
            framed.alias(..frame_bytes).map_err(other)?,
        ),
        (first, framed.alias(frame_bytes..pair).map_err(other)?),
    ];
    copy(device, pairs.iter_mut().map(|(from, to)| (&*from, to)))?;
    let before: Signal<S::Coded, FramePair> = alias(&framed, ..coded * pair)?;
    let own: Signal<S::Coded, FramePair> = alias(&framed, pair..)?;
    let [early, late] = &encoder.downsample;
    let early = linear(device, early, &before).await?;
    let late = linear(device, late, &own).await?;
    add(device, &early, &late).await
}

/// Each live codec frame's sixteen codes: the semantic codebook's pick of the frames through its
/// projection, then each acoustic codebook's pick of the residual of the frames through theirs,
/// less each pick's entry.
async fn quantize<S: Span>(
    device: &mut Device,
    encoder: &Encoder,
    latents: &Signal<S::Coded, HiddenOut>,
    live: usize,
) -> io::Result<Vec<[i32; GROUPS]>> {
    let mut codes = vec![[0i32; GROUPS]; live];
    let [semantic, acoustic] = &encoder.projections;
    let projected: Signal<S::Coded, Code> = linear(device, semantic, latents).await?;
    let picks = pick::<S>(device, &encoder.codebooks[0], &projected).await?;
    for (frame, &code) in codes.iter_mut().zip(&picks) {
        frame[0] = code;
    }
    let projected: Signal<S::Coded, Code> = linear(device, acoustic, latents).await?;
    let mut residual: Wide<S::Coded, Code> = Function::new(&Cast::<f32>::default(), forward)
        .forward(device, (), &mut (), &projected)
        .await?;
    for group in 1..GROUPS {
        let rows: Signal<S::Coded, Code> = Function::new(&Cast::<bf16>::default(), forward)
            .forward(device, (), &mut (), &residual)
            .await?;
        let picks = pick::<S>(device, &encoder.codebooks[group], &rows).await?;
        for (frame, &code) in codes.iter_mut().zip(&picks) {
            frame[group] = code;
        }
        if group + 1 == GROUPS {
            break;
        }
        // The picked entries, gathered by the byte offsets of their rows, less from the residual.
        let row = Code::SIZE * 2;
        let offsets: Vec<u8> = (picks.iter())
            .flat_map(|&code| (((group * 2048 + code as usize) * row) as i32).to_le_bytes())
            .collect();
        let offsets: HbmTensor<i32, Chip, m![{ Symbol<S::Coded> }]> =
            alias(&upload(device, &offsets).await?, ..)?;
        let entries: Signal<S::Coded, Code> = Function::new(&Gather, forward)
            .forward(device, (), &mut (), (&encoder.entries, &offsets))
            .await?;
        let entries: Wide<S::Coded, Code> = Function::new(&Cast::<f32>::default(), forward)
            .forward(device, (), &mut (), &entries)
            .await?;
        residual = Function::new(&Subtract, forward)
            .forward(device, (), &mut (), (&residual, &entries))
            .await?;
    }
    Ok(codes)
}

/// Each codec frame's code of `codebook`: the first highest of its FP32 scores of `rows`, every
/// launch row's.
async fn pick<S: Span>(
    device: &mut Device,
    codebook: &Biased<Code, Entries>,
    rows: &Signal<S::Coded, Code>,
) -> io::Result<Vec<i32>> {
    let projection = Unrounded::<Code, Entries, Chip, _, _>(
        Affine::new(&codebook.weight, &codebook.bias),
        PhantomData,
    );
    let scores: Wide<S::Coded, Entries> = Function::new(&projection, forward)
        .forward(device, (), &mut (), rows)
        .await?;
    let host = scores
        .to_host::<m![{ Symbol<S::Coded> }, Entries]>(&mut device.pdma)
        .await
        .map_err(io::Error::other)?
        .into_vec();
    Ok((host.chunks(Entries::SIZE))
        .map(|row| {
            (0..row.len()).fold(0, |best, at| if row[at] > row[best] { at } else { best }) as i32
        })
        .collect())
}

async fn upload(device: &mut Device, bytes: &[u8]) -> io::Result<HbmBuffer> {
    (HostBufferSlice::from(bytes).to_hbm(&mut device.pdma).await).map_err(io::Error::other)
}

const _: () = assert!(FRAMES * 960 == SAMPLES && Long::SAMPLES == SAMPLES);
