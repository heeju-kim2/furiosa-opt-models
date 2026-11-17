//! The speech decoder's front on the device: a launch's codes through `quantizer`, `pre_conv` and
//! `pre_transformer` into the latent rows [`super::Waveform`] decodes.
//!
//! - A frame's sixteen codes pick one entry of each codebook. Every codebook's entries lie in one
//!   table, so one gather by the byte offsets the host plans picks a frame's sixteen entries as one
//!   row, and one projection, the codebooks' `output_proj` side by side, sums them.
//! - `pre_conv` is three tap projections of a framed buffer, as the waveform's convolutions are.
//! - The transformer attends each frame's sliding window of [`SLIDING`] frames with products of
//!   every head at once: the scores against the keys laid out block-diagonally, one head a block,
//!   each score's exponential under the window's additive mask, then the products of the weights
//!   with the block-diagonal values and with block-diagonal ones, whose quotient is each head's
//!   softmax-weighted values. A row's scores stay well inside FP32's exponential, so no maximum is
//!   subtracted.
//! - Rotary embedding is `x cos + rotate_half(x) sin`, `rotate_half` a second projection of the
//!   permuted rows; `1 / sqrt(64)` and each layer scale fold into the weights.

use furiosa_opt_std::prelude::*;

use super::{Biased, Frames, LEAD, Latent, floats, taps, zeros};
use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::ops::linear::Linear;
use crate::ops::normalization::RmsNorm;
use crate::tensor::axis;
use crate::tensor::form::Weight;
use crate::tensor::parallel::Chip;

axes![
    Code = 256,
    Entries = 32768,
    Picked = 4096,
    Quantized = 512,
    Narrow = 512,
    Attention = 1024,
    Keyed = 2048,
    Inner = 1024
];

/// The codebooks of a frame's codes, and the entries of each.
pub const GROUPS: usize = 16;
pub const ENTRIES: usize = 2048;

/// The transformer's heads, their width and its layers.
pub const HEADS: usize = 16;
pub const HEAD: usize = 64;
pub const LAYERS: usize = 8;

/// Frames a row attends: itself and the 71 before it.
pub const SLIDING: usize = 72;

/// A launch's picks: [`GROUPS`] codebook entries a frame, frame-major.
pub type Picks = axis::Rows<2048>;

/// The additive mask of a score outside its row's window, whose exponential is zero.
const MASKED: f32 = -30000.0;

/// The transformer's `rms_norm_eps`.
const EPSILON: usize = 1e-5_f32.to_bits() as usize;

/// `Qwen3TTSTokenizerV2DecoderRMSNorm` of the transformer's width.
pub type Norm = RmsNorm<Narrow, { EPSILON }, Chip, Weight<m![Narrow], Chip>>;

/// `Qwen3TTSTokenizerV2DecoderTransformerLayer`: pre-norm attention over the sliding window, then
/// a pre-norm SiLU-gated feed-forward, each scaled and added to its input.
#[derive(DeviceSend)]
pub struct Layer {
    pub input_layernorm: Norm,
    /// `q_proj` over 8, then its rows permuted as `rotate_half` reads them.
    pub q: [Linear<Narrow, Attention, Chip>; 2],
    /// `k_proj`, then its rows permuted as `rotate_half` reads them.
    pub k: [Linear<Narrow, Attention, Chip>; 2],
    /// `v_proj`'s rows, which a call projects by its normed rows into the values' transpose.
    pub v: HbmTensor<bf16, Chip, m![Attention, Narrow]>,
    /// Times `self_attn_layer_scale`.
    pub o_proj: Linear<Attention, Narrow, Chip>,
    pub post_attention_layernorm: Norm,
    pub gate: Linear<Narrow, Inner, Chip>,
    pub up: Linear<Narrow, Inner, Chip>,
    /// Times `mlp_layer_scale`.
    pub down: Linear<Inner, Narrow, Chip>,
}

/// The codebooks, `pre_conv` and the transformer, with the constants and buffers a launch reads.
#[derive(DeviceSend)]
pub struct Front {
    /// Each codebook's entries, `embedding_sum / cluster_usage`, codebook-major.
    pub codebooks: HbmTensor<bf16, Chip, m![Entries, Code]>,
    /// The first codebook's `output_proj`, then the others', each of a frame's entries.
    pub quantize: Linear<Picked, Quantized, Chip>,
    pub pre_conv: [Biased<Quantized, Latent>; 3],
    pub input_proj: Biased<Latent, Narrow>,
    pub layers: [Layer; LAYERS],
    pub norm: Norm,
    pub output_proj: Biased<Narrow, Latent>,
    /// The cosine and the sine of each feature's angle at its row's frame.
    pub rotation: [FrameHeads; 2],
    /// Each head's mask of a launch's keys: one at the head's features, zero elsewhere, which a
    /// call multiplies its keys by into the head's block of the block-diagonal keys.
    pub keys: [FrameHeads; HEADS],
    /// One where a feature and a key row are of one head, zero elsewhere.
    pub values: HbmTensor<bf16, Chip, m![Attention, Keyed]>,
    /// Each head's sum of a row's weights, in every feature of the head.
    pub masses: Linear<Keyed, Attention, Chip>,
    /// Zero where a row sees a key of a head, [`MASKED`] elsewhere.
    pub window: HbmTensor<bf16, Chip, m![Frames, Keyed]>,
    /// `pre_conv`'s framed input.
    pub framed: HbmBuffer,
    /// Every head's masked rotated keys of a launch, one block a head, then a copy for every
    /// head of its normed rows.
    pub repeated: [HbmBuffer; 2],
}

/// A launch's rows of every head's features.
pub type FrameHeads = HbmTensor<bf16, Chip, m![Frames, Attention]>;

impl Front {
    /// `quantizer`, `pre_conv` and `pre_transformer` at the current name; the quantizer's
    /// `input_proj`, which only encoding reads, ignored.
    pub(super) fn read(cx: &mut Context<'_>, from: &Checkpoint) -> Result<Self, LoadError> {
        let (code, quantized) = (Code::SIZE, Quantized::SIZE);
        let mut entries = Vec::with_capacity(Entries::SIZE * code);
        for group in 0..GROUPS {
            let at = match group {
                0 => "quantizer.rvq_first.vq.layers.0._codebook".to_string(),
                _ => format!("quantizer.rvq_rest.vq.layers.{}._codebook", group - 1),
            };
            let usage = floats(cx, from, &format!("{at}.cluster_usage"), &[ENTRIES])?;
            let sums = floats(cx, from, &format!("{at}.embedding_sum"), &[ENTRIES, code])?;
            entries.extend(
                (0..ENTRIES * code).map(|at| bf16::from_f32(sums[at] / usage[at / code].max(1e-5))),
            );
        }
        let codebooks =
            HbmTensor::load(cx, HostTensor::<bf16, m![Entries, Code]>::from_vec(entries))?;
        let first = floats(
            cx,
            from,
            "quantizer.rvq_first.output_proj.weight",
            &[quantized, code, 1],
        )?;
        let rest = floats(
            cx,
            from,
            "quantizer.rvq_rest.output_proj.weight",
            &[quantized, code, 1],
        )?;
        for ignored in [
            "quantizer.rvq_first.input_proj",
            "quantizer.rvq_rest.input_proj",
        ] {
            cx.ignore(ignored);
        }
        let quantize = linear::<Picked, Quantized>(cx, |out, input| {
            let (group, input) = (input / code, input % code);
            let stored = if group == 0 { &first } else { &rest };
            stored[out * code + input]
        })?;
        let latent = Latent::SIZE;
        let weight = floats(cx, from, "pre_conv.conv.weight", &[latent, quantized, 3])?;
        let bias = floats(cx, from, "pre_conv.conv.bias", &[latent])?;
        let pre_conv = taps::<3, Quantized, Latent>(
            cx,
            |tap, out, input| Some(weight[(out * quantized + input) * 3 + tap]),
            |out| Some(bias[out]),
        )?;
        let narrow = Narrow::SIZE;
        let at = "pre_transformer";
        let weight = floats(
            cx,
            from,
            &format!("{at}.input_proj.weight"),
            &[narrow, latent],
        )?;
        let bias = floats(cx, from, &format!("{at}.input_proj.bias"), &[narrow])?;
        let [input_proj] = taps::<1, Latent, Narrow>(
            cx,
            |_, out, input| Some(weight[out * latent + input]),
            |out| Some(bias[out]),
        )?;
        let mut layers = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            layers.push(Layer::read(cx, from, &format!("{at}.layers.{layer}"))?);
        }
        let norm = norm(cx, from, &format!("{at}.norm.weight"))?;
        let weight = floats(
            cx,
            from,
            &format!("{at}.output_proj.weight"),
            &[latent, narrow],
        )?;
        let bias = floats(cx, from, &format!("{at}.output_proj.bias"), &[latent])?;
        let [output_proj] = taps::<1, Narrow, Latent>(
            cx,
            |_, out, input| Some(weight[out * narrow + input]),
            |out| Some(bias[out]),
        )?;
        // Feature `(h, d)` of a frame turns at `10000^(-2 (d mod 32) / 64)` radians a frame.
        let angle = |at: usize| {
            let (frame, feature) = (at / Attention::SIZE, at % Attention::SIZE);
            let pair = feature % HEAD % (HEAD / 2);
            frame as f64 * 10000f64.powf(-2.0 * pair as f64 / HEAD as f64)
        };
        let table = |turn: fn(f64) -> f64| {
            let values = (0..Frames::SIZE * Attention::SIZE)
                .map(|at| bf16::from_f32(turn(angle(at)) as f32))
                .collect::<Vec<_>>();
            HostTensor::<bf16, m![Frames, Attention]>::from_vec(values)
        };
        let rotation = [
            HbmTensor::load(cx, table(f64::cos))?,
            HbmTensor::load(cx, table(f64::sin))?,
        ];
        let one = |same: bool| bf16::from_f32(if same { 1.0 } else { 0.0 });
        let keyed = Keyed::SIZE / HEADS;
        let mut keys = Vec::with_capacity(HEADS);
        for head in 0..HEADS {
            let mask = (0..Frames::SIZE * Attention::SIZE)
                .map(|at| one(at % Attention::SIZE / HEAD == head))
                .collect::<Vec<_>>();
            keys.push(HbmTensor::load(
                cx,
                HostTensor::<bf16, m![Frames, Attention]>::from_vec(mask),
            )?);
        }
        let block = |at: usize| one(at / Keyed::SIZE / HEAD == at % Keyed::SIZE / keyed);
        let values = (0..Attention::SIZE * Keyed::SIZE)
            .map(block)
            .collect::<Vec<_>>();
        let values = HbmTensor::load(
            cx,
            HostTensor::<bf16, m![Attention, Keyed]>::from_vec(values),
        )?;
        let masses =
            linear::<Keyed, Attention>(cx, |out, input| f32::from(out / HEAD == input / keyed))?;
        let window = (0..Frames::SIZE * Keyed::SIZE)
            .map(|at| {
                let (row, key) = (at / Keyed::SIZE, at % Keyed::SIZE % keyed);
                let seen = key <= row && row - key < SLIDING;
                bf16::from_f32(if seen { 0.0 } else { MASKED })
            })
            .collect::<Vec<_>>();
        let window = HbmTensor::load(cx, HostTensor::<bf16, m![Frames, Keyed]>::from_vec(window))?;
        Ok(Self {
            codebooks,
            quantize,
            pre_conv,
            input_proj,
            layers: layers
                .try_into()
                .unwrap_or_else(|_| unreachable!("a layer an index")),
            norm,
            output_proj,
            rotation,
            keys: keys
                .try_into()
                .unwrap_or_else(|_| unreachable!("a mask a head")),
            values,
            masses,
            window,
            framed: zeros::<axis::Rows<{ LEAD + 128 }>, Quantized>(cx)?,
            repeated: [zeros::<Keyed, Attention>(cx)?, zeros::<Keyed, Narrow>(cx)?],
        })
    }
}

impl Layer {
    fn read(cx: &mut Context<'_>, from: &Checkpoint, at: &str) -> Result<Self, LoadError> {
        let (narrow, attention, inner) = (Narrow::SIZE, Attention::SIZE, Inner::SIZE);
        let input_layernorm = norm(cx, from, &format!("{at}.input_layernorm.weight"))?;
        let mut rotated = |name: &str, scale: f32| -> Result<_, LoadError> {
            let weight = floats(
                cx,
                from,
                &format!("{at}.self_attn.{name}.weight"),
                &[attention, narrow],
            )?;
            let own =
                linear::<Narrow, Attention>(cx, |out, input| scale * weight[out * narrow + input])?;
            // `rotate_half` of feature `(h, d)`: `-x(h, d + 32)` in the first half, `x(h, d - 32)`
            // in the second.
            let turned = linear::<Narrow, Attention>(cx, |out, input| {
                let (head, feature) = (out / HEAD, out % HEAD);
                let half = HEAD / 2;
                let (partner, sign) = match feature < half {
                    true => (feature + half, -1.0),
                    false => (feature - half, 1.0),
                };
                sign * scale * weight[(head * HEAD + partner) * narrow + input]
            })?;
            Ok([own, turned])
        };
        let q = rotated("q_proj", 1.0 / (HEAD as f32).sqrt())?;
        let k = rotated("k_proj", 1.0)?;
        let weight = floats(
            cx,
            from,
            &format!("{at}.self_attn.v_proj.weight"),
            &[attention, narrow],
        )?;
        let v = HbmTensor::load(
            cx,
            HostTensor::<bf16, m![Attention, Narrow]>::from_vec(
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
            &[narrow],
        )?;
        let weight = floats(
            cx,
            from,
            &format!("{at}.self_attn.o_proj.weight"),
            &[narrow, attention],
        )?;
        let o_proj = linear::<Attention, Narrow>(cx, |out, input| {
            scale[out] * weight[out * attention + input]
        })?;
        let post_attention_layernorm =
            norm(cx, from, &format!("{at}.post_attention_layernorm.weight"))?;
        let mut feed = |name: &str| -> Result<_, LoadError> {
            let weight = floats(
                cx,
                from,
                &format!("{at}.mlp.{name}.weight"),
                &[inner, narrow],
            )?;
            linear::<Narrow, Inner>(cx, |out, input| weight[out * narrow + input])
        };
        let gate = feed("gate_proj")?;
        let up = feed("up_proj")?;
        let scale = floats(cx, from, &format!("{at}.mlp_layer_scale.scale"), &[narrow])?;
        let weight = floats(
            cx,
            from,
            &format!("{at}.mlp.down_proj.weight"),
            &[narrow, inner],
        )?;
        let down =
            linear::<Inner, Narrow>(cx, |out, input| scale[out] * weight[out * inner + input])?;
        Ok(Self {
            input_layernorm,
            q,
            k,
            v,
            o_proj,
            post_attention_layernorm,
            gate,
            up,
            down,
        })
    }
}

/// The norm whose weight is at `path`.
fn norm(cx: &mut Context<'_>, from: &Checkpoint, path: &str) -> Result<Norm, LoadError> {
    let weight = floats(cx, from, path, &[Narrow::SIZE])?;
    let weight = HostTensor::<bf16, m![Narrow]>::from_vec(
        weight
            .iter()
            .map(|&value| bf16::from_f32(value))
            .collect::<Vec<_>>(),
    );
    Ok(RmsNorm::new(Weight::new(HbmTensor::load(cx, weight)?)))
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
