//! Qwen3-TTS's speech decoder on the device: a launch's codec frames through the codebooks,
//! `pre_conv` and the transformer into latent rows, then through the upsamplers, the convolution
//! and the blocks into contiguous 24 kHz samples, each convolution a sum of tap projections of
//! windows of a framed buffer.
//!
//! - The host plans the byte offsets of each frame's codebook entries ([`offsets`]); the device
//!   reads no code.

use crate::{AsyncModule, Function, forward};
use furiosa_opt_models::load::{LoadError, Part};
use furiosa_opt_models::models::decoder::attention::qwen3_tts::codec::front::{
    Attention, Code, ENTRIES, GROUPS, Keyed, Layer, Narrow, Picked, Picks, SLIDING,
};
use furiosa_opt_models::models::decoder::attention::qwen3_tts::codec::{
    BLOCK, Biased, Block, Blocks, Code2Wav, Codec, ConvNeXt, Convolution, DILATIONS, FRAMES, Fifth,
    Fifth16, Fourth4, Frames, Front, LEAD, LEADS, Latent, Latents, Phased, Samples, Snake,
    SnakeBeta, TAPS, Third2, Twice, Unit, Upsample, Waveform, Window, row_dilation,
};
use furiosa_opt_models::ops::index::copy;
use furiosa_opt_models::ops::index::gather::Gather;
use furiosa_opt_models::ops::linear::Linear;
use furiosa_opt_models::ops::pointwise::activation::{Exp, Gelu, Silu};
use furiosa_opt_models::ops::pointwise::multiply::{Divide, Multiply};
use furiosa_opt_models::ops::pointwise::{add::Add, clamp::Clamp};
use furiosa_opt_models::tensor::axis;
use furiosa_opt_models::tensor::form::Weight;
use furiosa_opt_models::tensor::memory::alias;
use furiosa_opt_models::tensor::parallel::Chip;
use furiosa_opt_models::tensor::scalar::F32;
use furiosa_opt_std::prelude::*;
use std::io;

pub use furiosa_opt_models::models::decoder::attention::qwen3_tts::codec::front::Quantized;
pub use furiosa_opt_models::models::decoder::attention::qwen3_tts::codec::{
    First, Fourth, LEADS as Leads, Rows1, Rows2, Rows3, Rows4, SAMPLES, Second, Third,
};

/// Rows of `W` channels, one a sample.
pub type Signal<R, W> = HbmTensor<bf16, Chip, m![R, W]>;

/// A launch's samples, [`BLOCK`] a row, each within [-1, 1].
pub type Wave = Signal<Blocks, Samples>;

/// The second block's output rows two samples a row, the third's four, and the last's sixteen.
pub type Pairs = axis::Rows<{ Rows2::SIZE / 2 }>;
pub type Quads = axis::Rows<{ Rows3::SIZE / 4 }>;
pub type Sixteens = axis::Rows<{ Rows4::SIZE / 16 }>;

/// `torch.clamp(-1, 1)`.
type Unit1 = Clamp<F32<{ (-1.0_f32).to_bits() as usize }>, F32<{ 1.0_f32.to_bits() as usize }>>;

/// `rows` projected by `projection`.
pub(super) async fn project<R, In, Out>(
    device: &mut Device,
    projection: &Biased<In, Out>,
    rows: &Signal<R, In>,
) -> io::Result<Signal<R, Out>>
where
    R: AxisName + Send + Sync,
    In: AxisName + Send + Sync,
    Out: AxisName + Send + Sync,
{
    Function::new(projection, forward)
        .forward(device, (), &mut (), rows)
        .await
}

pub(super) async fn add<R, W>(
    device: &mut Device,
    a: &Signal<R, W>,
    b: &Signal<R, W>,
) -> io::Result<Signal<R, W>>
where
    R: AxisName + Send + Sync,
    W: AxisName + Send + Sync,
{
    Function::new(&Add, forward)
        .forward(device, (), &mut (), (a, b))
        .await
}

/// SnakeBeta: `x` plus its periodic term, one launch.
async fn snake<R, W>(
    device: &mut Device,
    snake: &Snake<W>,
    x: &Signal<R, W>,
) -> io::Result<Signal<R, W>>
where
    R: AxisName + Send + Sync,
    W: AxisName + Send + Sync,
{
    Function::new(SnakeBeta(snake), forward)
        .forward(device, (), &mut (), x)
        .await
}

/// SnakeBeta of `x` launched into `framed`'s rows from row `start` on, the window it fills: the
/// convolution after reads its taps' windows of `framed` with no copy between.
fn snake_framed<R, W>(
    device: &mut Device,
    snake: &Snake<W>,
    x: &Signal<R, W>,
    (framed, start): (&HbmBuffer, usize),
) -> io::Result<Signal<R, W>>
where
    R: AxisName + Send + Sync,
    W: AxisName + Send + Sync,
{
    let mut activated = window(framed, start)?;
    Function::new(SnakeBeta(snake), forward).forward_into(
        device,
        (),
        &mut (),
        x,
        &mut activated,
    )?;
    Ok(activated)
}

/// `tensor`'s storage as rows of another shape.
pub fn recast<A: M, B: M>(
    tensor: HbmTensor<bf16, Chip, A>,
) -> io::Result<HbmTensor<bf16, Chip, B>> {
    alias(&tensor.into_buffer(), ..)
}

/// `rows` of `framed` from row `start` on, as a signal.
pub(super) fn window<R: AxisName, W: AxisName>(
    framed: &HbmBuffer,
    start: usize,
) -> io::Result<Signal<R, W>> {
    let row = W::SIZE * size_of::<bf16>();
    alias(framed, start * row..(start + R::SIZE) * row)
}

/// `signal` copied into `framed` from row `start` on, and the copy as a signal.
pub(super) fn frame<R: AxisName, W: AxisName>(
    device: &mut Device,
    framed: &HbmBuffer,
    start: usize,
    signal: Signal<R, W>,
) -> io::Result<Signal<R, W>> {
    let row = W::SIZE * size_of::<bf16>();
    let source = signal.into_buffer();
    let mut target = framed
        .alias(start * row..(start + R::SIZE) * row)
        .map_err(io::Error::other)?;
    copy(device, [(&source, &mut target)])?;
    window(framed, start)
}

/// The sum of each of `taps`' projection of its window of `framed`, tap `k`'s from row
/// `starts[k]`, one launch.
pub(super) async fn convolve_windows<R, In, Out, const N: usize>(
    device: &mut Device,
    framed: &HbmBuffer,
    taps: &[Biased<In, Out>; N],
    starts: [usize; N],
) -> io::Result<Signal<R, Out>>
where
    R: AxisName + Send + Sync,
    In: AxisName + Send + Sync,
    Out: AxisName + Send + Sync,
{
    let windows: [Signal<R, In>; N] = (starts.iter())
        .map(|&start| window::<R, In>(framed, start))
        .collect::<io::Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| io::Error::other("a window a tap"))?;
    Function::new(Convolution(taps), forward)
        .forward(device, (), &mut (), &windows)
        .await
}

/// The causal convolution of the signal framed in `framed` after `lead` rows by `taps`, `apart`
/// rows apart: tap `k` of `N` projects the window `(N - 1 - k) apart` rows before the signal's.
async fn causal<R, In, Out, const N: usize>(
    device: &mut Device,
    framed: &HbmBuffer,
    taps: &[Biased<In, Out>; N],
    (lead, apart): (usize, usize),
) -> io::Result<Signal<R, Out>>
where
    R: AxisName + Send + Sync,
    In: AxisName + Send + Sync,
    Out: AxisName + Send + Sync,
{
    let starts = core::array::from_fn(|tap| lead - (N - 1 - tap) * apart);
    convolve_windows::<R, In, Out, N>(device, framed, taps, starts).await
}

/// `Qwen3TTSTokenizerV2ConvNeXtBlock` of `x`, framed in `framed`.
async fn convnext<R>(
    device: &mut Device,
    block: &ConvNeXt,
    framed: &HbmBuffer,
    x: Signal<R, Latent>,
) -> io::Result<Signal<R, Latent>>
where
    R: AxisName + Send + Sync,
{
    let x = frame(device, framed, LEAD, x)?;
    let filtered: Signal<R, Latent> =
        recast(causal::<R, _, _, TAPS>(device, framed, &block.dwconv, (LEAD, 1)).await?)?;
    let normed: Signal<R, Latent> = Function::new(&block.norm, forward)
        .forward(device, (), &mut (), &filtered)
        .await?;
    let wide = project(device, &block.pwconv1, &normed).await?;
    let wide = Function::new(&Gelu, forward)
        .forward(device, (), &mut (), &wide)
        .await?;
    let narrow = project(device, &block.pwconv2, &wide).await?;
    add(device, &x, &narrow).await
}

/// An upsampler of `x`: both phases of each row, read as twice the rows, then its ConvNeXt block,
/// framed in `framed`.
pub async fn upsample<R, Doubled>(
    device: &mut Device,
    upsample: &Upsample,
    framed: &HbmBuffer,
    x: &Signal<R, Latent>,
) -> io::Result<Signal<Doubled, Latent>>
where
    R: AxisName + Send + Sync,
    Doubled: AxisName + Send + Sync,
{
    const { assert!(Doubled::SIZE == 2 * R::SIZE) };
    let phased = project(device, &upsample.transpose, x).await?;
    convnext(device, &upsample.block, framed, recast(phased)?).await
}

/// A residual unit of `x`, rows of `C::PHASES` samples, its taps `dilation` samples apart, framed
/// in `framed` after `lead` samples.
async fn unit<C, const N: usize>(
    device: &mut Device,
    unit: &Unit<C, N>,
    (framed, lead): (&HbmBuffer, usize),
    dilation: usize,
    x: &Signal<C::Rows, C>,
) -> io::Result<Signal<C::Rows, C>>
where
    C: Phased + Send + Sync,
    C::Out: Send + Sync,
    C::Rows: Send + Sync,
{
    let rows = lead / C::PHASES;
    snake_framed(device, &unit.act1, x, (framed, rows))?;
    let apart = row_dilation(C::PHASES, dilation);
    let filtered: Signal<C::Rows, C> =
        recast(causal::<C::Rows, _, _, N>(device, framed, &unit.conv1, (rows, apart)).await?)?;
    let activated = snake(device, &unit.act2, &filtered).await?;
    let projected: Signal<C::Rows, C> =
        recast(project::<C::Rows, _, _>(device, &unit.conv2, &activated).await?)?;
    add(device, x, &projected).await
}

/// A block of `x`: SnakeBeta framed in `input` after its lead of samples, the transposed
/// convolution of each row and the row before into its phases, then the residual units over rows
/// of `W`'s phases framed in `output` after its lead, read back as the output's rows.
pub async fn block<RIn, ROut, In, Out, P, W, const A: usize, const B: usize, const C: usize>(
    device: &mut Device,
    block: &Block<In, Out, P, W, A, B, C>,
    ((input, lead), output): ((&HbmBuffer, usize), (&HbmBuffer, usize)),
    x: &Signal<RIn, In>,
) -> io::Result<Signal<ROut, Out>>
where
    RIn: AxisName + Send + Sync,
    ROut: AxisName + Send + Sync,
    In: AxisName + Send + Sync,
    Out: AxisName + Send + Sync,
    P: AxisName + Send + Sync,
    W: Phased + Send + Sync,
    W::Out: Send + Sync,
    W::Rows: Send + Sync,
{
    const { assert!(RIn::SIZE * P::SIZE == ROut::SIZE * Out::SIZE) };
    const { assert!(ROut::SIZE * Out::SIZE == W::Rows::SIZE * W::SIZE) };
    snake_framed(device, &block.snake, x, (input, lead))?;
    let phased: Signal<RIn, P> =
        convolve_windows::<RIn, In, P, 2>(device, input, &block.transpose, [lead, lead - 1])
            .await?;
    let (first, second, third) = &block.units;
    let x: Signal<W::Rows, W> = recast(phased)?;
    let x = unit(device, first, output, DILATIONS[0], &x).await?;
    let x = unit(device, second, output, DILATIONS[1], &x).await?;
    let x = unit(device, third, output, DILATIONS[2], &x).await?;
    recast(x)
}

/// The decoder's first convolution of the second upsampler's rows `x`, framed in `framed`.
pub async fn convolve(
    device: &mut Device,
    waveform: &Waveform,
    x: Signal<Latents, Latent>,
) -> io::Result<Signal<Latents, First>> {
    let framed = &waveform.framed.latents;
    frame(device, framed, LEAD, x)?;
    causal::<Latents, _, _, TAPS>(device, framed, &waveform.conv, (LEAD, 1)).await
}

/// The last SnakeBeta and convolution of the last block's rows `x`, clamped: SnakeBeta over rows of
/// 16 samples, then each output row the projection of its block of input rows and of the block
/// before.
pub async fn output(
    device: &mut Device,
    waveform: &Waveform,
    x: Signal<Rows4, Fifth>,
) -> io::Result<Wave> {
    const { assert!(BLOCK.is_multiple_of(16)) };
    let framed = &waveform.framed.output;
    let x: Signal<Sixteens, Fifth16> = recast(x)?;
    snake_framed(device, &waveform.snake, &x, (framed, BLOCK / 16))?;
    let wave: Wave =
        convolve_windows::<Blocks, Window, Samples, 2>(device, framed, &waveform.output, [1, 0])
            .await?;
    Function::new(&Unit1::default(), forward)
        .forward(device, (), &mut (), &wave)
        .await
}

/// A launch's latent rows `x`, one a codec frame, as samples.
pub async fn decode(
    device: &mut Device,
    waveform: &Waveform,
    x: &Signal<Frames, Latent>,
) -> io::Result<Wave> {
    let framed = &waveform.framed;
    let [first, second] = &waveform.upsample;
    let x: Signal<Twice, Latent> = upsample(device, first, &framed.twice, x).await?;
    let x: Signal<Latents, Latent> = upsample(device, second, &framed.latents, &x).await?;
    let x = convolve(device, waveform, x).await?;
    let x: Signal<Rows1, Second> = block(
        device,
        &waveform.block1,
        ((&framed.first, LEAD), (&framed.second, LEADS[0])),
        &x,
    )
    .await?;
    let x: Signal<Rows2, Third> = block(
        device,
        &waveform.block2,
        ((&framed.second, LEADS[0]), (&framed.third, LEADS[1])),
        &x,
    )
    .await?;
    // The later blocks read rows of two and of four samples, as their units' rows hold.
    let x: Signal<Pairs, Third2> = recast(x)?;
    let x: Signal<Rows3, Fourth> = block(
        device,
        &waveform.block3,
        ((&framed.third, LEADS[1] / 2), (&framed.fourth, LEADS[2])),
        &x,
    )
    .await?;
    let x: Signal<Quads, Fourth4> = recast(x)?;
    let x: Signal<Rows4, Fifth> = block(
        device,
        &waveform.block4,
        ((&framed.fourth, LEADS[2] / 4), (&framed.fifth, LEADS[3])),
        &x,
    )
    .await?;
    output(device, waveform, x).await
}

/// `rows` projected by `linear`, whose weight may be a call's own rows.
pub(super) async fn linear<R, In, Out, T>(
    device: &mut Device,
    linear: &Linear<In, Out, Chip, T>,
    rows: &Signal<R, In>,
) -> io::Result<Signal<R, Out>>
where
    R: AxisName + Send + Sync,
    In: AxisName + Send + Sync,
    Out: AxisName + Send + Sync,
    Linear<In, Out, Chip, T>: DeviceSend + Sync,
    T: core::borrow::Borrow<HbmTensor<bf16, Chip, m![Out, In]>>,
{
    Function::new(linear, forward)
        .forward(device, (), &mut (), rows)
        .await
}

/// The elementwise `op` of `a` and `b`.
pub(super) async fn combine<R, W, Op>(
    device: &mut Device,
    op: &Op,
    a: &Signal<R, W>,
    b: &Signal<R, W>,
) -> io::Result<Signal<R, W>>
where
    R: AxisName + Send + Sync,
    W: AxisName + Send + Sync,
    Op: DeviceSend
        + Sync
        + for<'x> furiosa_opt_models::Op<(&'x Signal<R, W>, &'x Signal<R, W>), Output = Signal<R, W>>,
{
    Function::new(op, forward)
        .forward(device, (), &mut (), (a, b))
        .await
}

/// The byte offsets a launch's picks gather, frame-major: frame `t`'s entry of codebook `g` lies at
/// row `2048 g + codes[t][g]` of the codebooks. The launch's frames past `codes` pick each first
/// entry.
pub fn offsets(codes: &[[i32; GROUPS]]) -> io::Result<Vec<i32>> {
    let invalid = |reason: &str| io::Error::new(io::ErrorKind::InvalidInput, reason.to_string());
    if codes.len() > FRAMES {
        return Err(invalid("a launch holds 128 frames"));
    }
    let row = Code::SIZE * size_of::<bf16>();
    let mut offsets = vec![0; FRAMES * GROUPS];
    for (frame, entries) in codes.iter().enumerate() {
        for (group, &code) in entries.iter().enumerate() {
            let code = usize::try_from(code)
                .ok()
                .filter(|&code| code < ENTRIES)
                .ok_or_else(|| invalid("a code lies outside its codebook"))?;
            let at = (group * ENTRIES + code) * row;
            offsets[frame * GROUPS + group] =
                i32::try_from(at).map_err(|_| invalid("an offset exceeds I32"))?;
        }
    }
    for (at, offset) in offsets.iter_mut().enumerate().skip(codes.len() * GROUPS) {
        *offset = ((at % GROUPS) * ENTRIES * row) as i32;
    }
    Ok(offsets)
}

/// Each frame's entries of the codebooks the `offsets` pick, summed through `output_proj`.
pub async fn quantize(
    device: &mut Device,
    front: &Front,
    offsets: &HbmTensor<i32, Chip, m![Picks]>,
) -> io::Result<Signal<Frames, Quantized>> {
    let picked: HbmTensor<bf16, Chip, m![Picks, Code]> = Function::new(&Gather, forward)
        .forward(device, (), &mut (), (&front.codebooks, offsets))
        .await?;
    let picked: Signal<Frames, Picked> = recast(picked)?;
    linear(device, &front.quantize, &picked).await
}

/// `pre_conv` of the quantized rows `x`.
pub async fn convolve_codes(
    device: &mut Device,
    front: &Front,
    x: Signal<Frames, Quantized>,
) -> io::Result<Signal<Frames, Latent>> {
    frame(device, &front.framed, LEAD, x)?;
    causal::<Frames, _, _, 3>(device, &front.framed, &front.pre_conv, (LEAD, 1)).await
}

/// `x cos + rotate_half(x) sin` of the projections `pair` of `x`.
async fn rotate(
    device: &mut Device,
    front: &Front,
    pair: &[Linear<Narrow, Attention, Chip>; 2],
    x: &Signal<Frames, Narrow>,
) -> io::Result<Signal<Frames, Attention>> {
    let [own, turned] = pair;
    let [cos, sin] = &front.rotation;
    let own = linear(device, own, x).await?;
    let turned = linear(device, turned, x).await?;
    let own = combine(device, &Multiply, &own, cos).await?;
    let turned = combine(device, &Multiply, &turned, sin).await?;
    add(device, &own, &turned).await
}

/// The rows `keys` times each head's mask of `masks`, the head's block of `buffer`, one after
/// another: the block-diagonal keys, a launch a head.
pub(super) fn masked<R, W, Stacked, const HEADS: usize>(
    device: &mut Device,
    masks: &[Signal<R, W>; HEADS],
    keys: &HbmBuffer,
    buffer: &HbmBuffer,
) -> io::Result<Signal<Stacked, W>>
where
    R: AxisName + Send + Sync,
    W: AxisName + Send + Sync,
    Stacked: AxisName,
{
    const { assert!(Stacked::SIZE == HEADS * R::SIZE) };
    let bytes = R::SIZE * W::SIZE * size_of::<bf16>();
    let keys: Signal<R, W> = alias(keys, ..)?;
    let function = Function::new(&Multiply, forward);
    for (head, mask) in masks.iter().enumerate() {
        let mut block: Signal<R, W> = alias(buffer, head * bytes..(head + 1) * bytes)?;
        function.forward_into(device, (), &mut (), (&keys, mask), &mut block)?;
    }
    alias(buffer, ..)
}

/// `norm` of `x` launched into each block of `buffer` in turn, `Stacked / R` blocks: as many copies
/// of the rows as a launch of the norm a block, with no copy between.
pub(super) fn norm_blocks<N, F, R, W, Stacked>(
    norm: &Function<N, F>,
    device: &mut Device,
    x: &Signal<R, W>,
    buffer: &HbmBuffer,
) -> io::Result<Signal<Stacked, W>>
where
    R: AxisName + Send + Sync,
    W: AxisName + Send + Sync,
    Stacked: AxisName,
    N: DeviceSend
        + for<'x> furiosa_opt_models::Module<&'x Signal<R, W>, (), &'x mut (), Output = Signal<R, W>>
        + Sync,
    F: Copy
        + Sync
        + for<'d, 'm, 'x, 's> DeviceFn<
            (&'d mut Device, &'m N, (), &'s mut (), &'x Signal<R, W>),
            Output = Signal<R, W>,
        >,
{
    const { assert!(Stacked::SIZE.is_multiple_of(R::SIZE)) };
    let bytes = R::SIZE * W::SIZE * size_of::<bf16>();
    for block in 0..Stacked::SIZE / R::SIZE {
        let mut rows: Signal<R, W> = alias(buffer, block * bytes..(block + 1) * bytes)?;
        norm.forward_into(device, (), &mut (), x, &mut rows)?;
    }
    alias(buffer, ..)
}

/// A transformer layer of `x`: the sliding window's attention as block-diagonal products of every
/// head, then the gated feed-forward.
async fn layer(
    device: &mut Device,
    front: &Front,
    layer: &Layer,
    x: Signal<Frames, Narrow>,
) -> io::Result<Signal<Frames, Narrow>> {
    let [keyed, rows] = &front.repeated;
    // The normed rows once for every head, each block of `rows` its own launch of the norm.
    let repeated: Signal<Keyed, Narrow> = norm_blocks(
        &Function::new(&layer.input_layernorm, forward),
        device,
        &x,
        rows,
    )?;
    let normed: Signal<Frames, Narrow> = alias(rows, ..Frames::SIZE * Narrow::SIZE * 2)?;
    let queries = rotate(device, front, &layer.q, &normed).await?;
    let keys = rotate(device, front, &layer.k, &normed)
        .await?
        .into_buffer();
    let keyed: Signal<Keyed, Attention> = masked(device, &front.keys, &keys, keyed)?;
    let scores: Signal<Frames, Keyed> =
        linear(device, &Linear::new(Weight::new(keyed)), &queries).await?;
    let scores = add(device, &scores, &front.window).await?;
    let weights = Function::new(&Exp, forward)
        .forward(device, (), &mut (), &scores)
        .await?;
    let values: Signal<Attention, Keyed> =
        linear(device, &Linear::new(Weight::new(repeated)), &layer.v).await?;
    let values = combine(device, &Multiply, &values, &front.values).await?;
    let sums: Signal<Frames, Attention> =
        linear(device, &Linear::new(Weight::new(values)), &weights).await?;
    let masses = linear(device, &front.masses, &weights).await?;
    let attended = combine(device, &Divide, &sums, &masses).await?;
    let projected = linear(device, &layer.o_proj, &attended).await?;
    let x = add(device, &x, &projected).await?;
    let normed: Signal<Frames, Narrow> = Function::new(&layer.post_attention_layernorm, forward)
        .forward(device, (), &mut (), &x)
        .await?;
    let gate = linear(device, &layer.gate, &normed).await?;
    let gate = Function::new(&Silu, forward)
        .forward(device, (), &mut (), &gate)
        .await?;
    let up = linear(device, &layer.up, &normed).await?;
    let gated = combine(device, &Multiply, &gate, &up).await?;
    let down = linear(device, &layer.down, &gated).await?;
    add(device, &x, &down).await
}

/// `pre_transformer` of the latent rows `x`: into its width, its layers and norm, and back.
pub async fn transform(
    device: &mut Device,
    front: &Front,
    x: &Signal<Frames, Latent>,
) -> io::Result<Signal<Frames, Latent>> {
    let mut x: Signal<Frames, Narrow> = project(device, &front.input_proj, x).await?;
    for layer_ in &front.layers {
        x = layer(device, front, layer_, x).await?;
    }
    let normed: Signal<Frames, Narrow> = Function::new(&front.norm, forward)
        .forward(device, (), &mut (), &x)
        .await?;
    project(device, &front.output_proj, &normed).await
}

/// A launch's codes, picked by `offsets`, as samples: [`SAMPLES`] a frame, the live frames'
/// first.
pub async fn synthesize(
    device: &mut Device,
    codec: &Codec,
    offsets: &HbmTensor<i32, Chip, m![Picks]>,
) -> io::Result<Wave> {
    let quantized = quantize(device, &codec.front, offsets).await?;
    let latent = convolve_codes(device, &codec.front, quantized).await?;
    let latent = transform(device, &codec.front, &latent).await?;
    decode(device, &codec.waveform, &latent).await
}

/// Frames before a launch's new frames that it decodes again for their samples' context: the
/// transformer's sliding window. A launch's first frames depend on much more than the 25 frames of
/// Transformers' `chunked_decode`, whose 300-frame chunks rarely need them: with 25 the frames
/// after a boundary keep a cosine of 0.2 to 0.97 of the whole decode's samples, with 72 above 0.95.
pub const CONTEXT: usize = SLIDING;

/// Codes of any count as their samples, [`SAMPLES`] a frame on the host: launches of the next
/// `FRAMES - CONTEXT` frames, each after the up to [`CONTEXT`] frames before them, whose samples
/// it leaves out, as Transformers' `chunked_decode` does with its context; up to [`FRAMES`] codes
/// decode in one launch, as the whole decode does.
pub async fn speak(
    device: &mut Device,
    codec: &Codec,
    codes: &[[i32; GROUPS]],
) -> io::Result<Vec<f32>> {
    let mut samples = Vec::with_capacity(codes.len() * SAMPLES);
    let mut start = 0;
    while start < codes.len() {
        let context = start.min(CONTEXT);
        let end = codes.len().min(start + FRAMES - context);
        let offsets = offsets(&codes[start - context..end])?;
        let bytes: Vec<u8> = offsets
            .iter()
            .flat_map(|offset| offset.to_le_bytes())
            .collect();
        let offsets: HbmTensor<i32, Chip, m![Picks]> = HostBufferSlice::from(&bytes[..])
            .to_hbm(&mut device.pdma)
            .await
            .map_err(io::Error::other)?
            .try_into()
            .map_err(io::Error::other)?;
        let wave = synthesize(device, codec, &offsets).await?;
        let wave = wave
            .to_host::<m![Blocks, Samples]>(&mut device.pdma)
            .await
            .map_err(io::Error::other)?
            .into_vec();
        let live = &wave[context * SAMPLES..(context + end - start) * SAMPLES];
        samples.extend(live.iter().map(|sample| sample.to_f32()));
        start = end;
    }
    Ok(samples)
}

/// The 12 Hz speech tokenizer's decoder serves a whole Qwen3-TTS checkpoint as its speech decoder.
impl crate::registry::Listed for Code2Wav {
    const ROLE: crate::registry::Role = crate::registry::Role::Vocoder;

    fn reads(part: Part) -> Result<(), LoadError> {
        match part {
            Part::Whole => Ok(()),
            Part::Encoder => Err(LoadError::Unsupported {
                name: "encoder".into(),
                reason: "a speech decoder holds no media encoder that loads alone".into(),
            }),
        }
    }

    fn served(
        self,
        _: furiosa_opt_models::ops::attention::full::paged::Tiling,
    ) -> crate::registry::Model {
        crate::registry::Model::Vocoder(Box::new(self))
    }
}

/// Frames of sixteen codes into 24 kHz samples, [`SAMPLES`] a frame, as [`speak`] decodes them.
impl super::Vocoder for Code2Wav {
    fn groups(&self) -> usize {
        GROUPS
    }

    fn entries(&self) -> usize {
        ENTRIES
    }

    fn samples(&self) -> usize {
        SAMPLES
    }

    fn rate(&self) -> usize {
        RATE
    }

    fn decode<'a>(
        &'a self,
        device: &'a mut Device,
        codes: &'a [i32],
    ) -> std::pin::Pin<Box<dyn Future<Output = io::Result<Vec<f32>>> + Send + 'a>> {
        Box::pin(async move {
            let (frames, []) = codes.as_chunks::<GROUPS>() else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "a frame holds sixteen codebook entries",
                ));
            };
            speak(device, &self.0, frames).await
        })
    }
}

/// The samples a second of the decoder's output holds.
pub const RATE: usize = 24000;
