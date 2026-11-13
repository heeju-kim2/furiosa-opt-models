//! SnakeBeta's periodic term over HBM rows, `c (1 - cos(2 alpha x))` with `c = 1 / (2 (beta +
//! 1e-9))`, which a caller adds to `x` for `x + sin^2(alpha x) / (beta + 1e-9)`.
//!
//! - The cosine takes arguments of at most one turn, so each row tile runs two programs: the first
//!   multiplies each feature by its `alpha / pi` in FP32 and stores the turns `u = alpha x / pi`
//!   in Q10.21 fixed point; the second keeps their fraction, a shift pair dropping the sign and
//!   integer bits, and scales it back into `[0, 2 pi)` before the cosine:
//!   `cos(2 alpha x) = cos(2 pi frac(u))`.
//! - A row tile and its parameters are laid out as a layer norm's are, each slice one contiguous
//!   part of a row's features and the per-feature parameters in its vector registers.

use core::f32::consts::TAU;

use furiosa_opt_std::prelude::*;

use crate::Op;
use crate::load::Load;
use crate::ops::normalization::{Tile, TilePart, TileRows};

/// The FP32 vectors of the features the second program keeps in its registers: the scale and its
/// negation; the first keeps one, the frequency.
const VECTORS: usize = 2;

/// Integer bits of the turns' fixed-point copy: `|alpha x / pi| < 2^9`.
const INTEGER: u32 = 10;

/// The shift that drops a fixed-point copy's sign and integer bits.
const DROPPED: u32 = INTEGER + 1;

/// SnakeBeta's per-feature parameters, from the checkpoint's log-scale `alpha` and `beta`.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct Periodic<W: AxisName, Chips: M = m![1]> {
    /// `alpha / pi`.
    pub frequency: HbmTensor<f32, Chips, m![W]>,
    /// `c = 1 / (2 (beta + 1e-9))`.
    pub scale: HbmTensor<f32, Chips, m![W]>,
    /// `-c`.
    pub weight: HbmTensor<f32, Chips, m![W]>,
}

impl<W: AxisName, Chips: M> Periodic<W, Chips> {
    /// The parameters of the log-scale `alpha` and `beta` of each feature, on the host.
    pub fn host(alpha: &[f32], beta: &[f32]) -> [HostTensor<f32, m![W]>; 3] {
        let scale = |b: &f32| (0.5 / (f64::from(*b).exp() + 1e-9)) as f32;
        let frequency = alpha
            .iter()
            .map(|a| (f64::from(*a).exp() / core::f64::consts::PI) as f32);
        [
            HostTensor::from_vec(frequency.collect::<Vec<_>>()),
            HostTensor::from_vec(beta.iter().map(scale).collect::<Vec<_>>()),
            HostTensor::from_vec(beta.iter().map(|b| -scale(b)).collect::<Vec<_>>()),
        ]
    }
}

/// Each row's periodic term, a tile of rows at a time.
impl<W: AxisName, R: AxisName, Chips: M> Op<&HbmTensor<bf16, Chips, m![R, W]>>
    for Periodic<W, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![R, W]>;

    fn forward(
        &self,
        device: &mut Device,
        input: &HbmTensor<bf16, Chips, m![R, W]>,
    ) -> Self::Output {
        type T<W> = Symbol<TileRows<W, VECTORS>>;
        type Part<W> = TilePart<W, VECTORS>;
        let tiles = const {
            assert!(
                R::SIZE.is_multiple_of(TileRows::<W, VECTORS>::SIZE),
                "rows come in whole tiles"
            );
            R::SIZE / TileRows::<W, VECTORS>::SIZE
        };
        let feature = |device: &mut Device, vector: &HbmTensor<f32, Chips, m![W]>| {
            let in_dm: Tile<f32, R, W, VECTORS, Chips> = vector.view().to_dm(&mut device.tdma);
            device
                .sub
                .begin(in_dm.view())
                .fetch::<m![{ Part<W> } / 8], m![W % 8]>()
                .collect::<m![{ Part<W> } / 8], m![W % 8]>()
                .to_vrf::<Part<W>>()
        };
        let frequency = feature(device, &self.frequency);
        let scale = feature(device, &self.scale);
        let weight = feature(device, &self.weight);
        let mut output = HbmTensor::new();
        for at in 0..tiles {
            let mut rows: Tile<bf16, R, W, VECTORS, Chips> = DmTensor::new();
            input
                .view()
                .tile::<m![R / ({ T<W> })], 1, m![R / ({ T<W> }) = 1 # (R / ({ T<W> })), R % ({ T<W> }), W]>(at)
                .to_dm_view(&mut device.tdma, rows.view_mut());
            let turns: Tile<i32, R, W, VECTORS, Chips> = device
                .main
                .begin(rows.view())
                .fetch::<m![{ Part<W> } / 8], m![W % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Part<W> } / 8], m![W % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ Part<W> } / 8, W % 8 / 4 % 2], m![W % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &frequency)
                .vector_widen_concat::<m![{ Part<W> } / 8], m![W % 8]>()
                .vector_fp_to_fxp(INTEGER)
                .vector_final()
                .commit_trim::<m![W % 8]>()
                .commit();
            let term: Tile<bf16, R, W, VECTORS, Chips> = device
                .main
                .begin(turns.view())
                .fetch::<m![{ Part<W> } / 8], m![W % 8]>()
                .collect::<m![{ Part<W> } / 8], m![W % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_fxp(FxpBinaryOp::LeftShift, DROPPED as i32)
                .vector_fxp(FxpBinaryOp::LogicRightShift, DROPPED as i32)
                .vector_fxp_to_fp(INTEGER)
                .vector_narrow_split::<m![{ Part<W> } / 8, W % 8 / 4 % 2], m![W % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), TAU)
                .vector_fp_unary(FpUnaryOp::Cos)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight)
                .vector_widen_concat::<m![{ Part<W> } / 8], m![W % 8]>()
                .vector_clip(ClipBinaryOpF32::Add, &scale)
                .vector_final()
                .cast::<bf16, m![W % 8 # 16]>()
                .commit_trim::<m![W % 8]>()
                .commit();
            term.view().to_hbm_view(
                &mut device.tdma,
                output
                    .view_mut()
                    .tile::<m![R / ({ T<W> })], 1, m![R / ({ T<W> }) = 1 #{!} (R / ({ T<W> })), R % ({ T<W> }), W]>(at),
            );
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use super::Periodic;
    use crate::Op;
    use crate::ops::normalization::tests::random;
    use crate::tests::DEVICE;

    axes![Rows = 256, Features = 768];

    #[device(chip = 1)]
    fn periodic(
        device: &mut Device,
        snake: &Periodic<Features>,
        input: &HbmTensor<bf16, m![1], m![Rows, Features]>,
    ) -> HbmTensor<bf16, m![1], m![Rows, Features]> {
        snake.forward(device, input)
    }

    /// The term of inputs over [-8, 8) and log-scale parameters over [-1, 1) is within a BF16
    /// step of `c (1 - cos(2 alpha x))` in FP64, and within 2^-12 of it near zero, arguments of
    /// up to 43 radians included.
    #[tokio::test(flavor = "current_thread")]
    async fn matches_the_periodic_term() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let x: Vec<bf16> = random(Rows::SIZE * Features::SIZE, 0x5eed_0001)
            .into_iter()
            .map(|value| bf16::from_f32(value.to_f32() * 8.0))
            .collect();
        let parameter = |seed| -> Vec<f32> {
            random(Features::SIZE, seed)
                .into_iter()
                .map(|v| v.to_f32())
                .collect()
        };
        let (alpha, beta) = (parameter(0x5eed_0002), parameter(0x5eed_0003));
        let [frequency, scale, weight] = Periodic::<Features>::host(&alpha, &beta);
        let snake = Periodic::<Features> {
            frequency: frequency.to_hbm(&mut device.pdma).await.unwrap(),
            scale: scale.to_hbm(&mut device.pdma).await.unwrap(),
            weight: weight.to_hbm(&mut device.pdma).await.unwrap(),
        };
        let input: HbmTensor<bf16, m![1], m![Rows, Features]> =
            HostTensor::<bf16, m![Rows, Features]>::from_vec(x.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let output = launch(periodic, (&mut device, &snake, &input))
            .await
            .unwrap()
            .to_host::<m![Rows, Features]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (at, (x, got)) in x.iter().zip(output).enumerate() {
            let feature = at % Features::SIZE;
            let a = f64::from(alpha[feature]).exp();
            let c = 0.5 / (f64::from(beta[feature]).exp() + 1e-9);
            let x = f64::from(x.to_f32());
            let want = c * (1.0 - (2.0 * a * x).cos());
            let got = f64::from(got.to_f32());
            let step = (want.abs().log2().floor() - 7.0).exp2();
            assert!(
                (got - want).abs() <= step.max(2f64.powi(-12)),
                "x {x}, alpha {a}: {got} vs {want}"
            );
        }
    }
}
