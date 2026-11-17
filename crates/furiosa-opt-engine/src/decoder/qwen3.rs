//! Qwen3's and Llama's listed roots as decoders: a call stages its requests, streams the base
//! model's body over every layer's own blocks into the call's rows, then keeps the rows its
//! requests keep.
//!
//! - The base model returns the kept rows' hidden values; the causal model runs its greedy head
//!   on them and returns their tokens; the sequence classifier runs its score head on them and
//!   returns their scores.
//! - A row of a BF16 or channel-scaled line attends its whole history as a work item of the size's
//!   attention, tiled as the call says.
//! - The catalog's launch model, in dense BF16 or block-scaled FP8, streams its whole causal tree
//!   in one launch: its host plans each row's position and cache slot, its work items and prompt
//!   chunks and its selected rows, and the launch returns the selected rows' tokens in kept order.
//!   Its prompt chunks attend apart from their history, so it refuses a fixed tiling.
//! - A call whose rows run with adapters passes each layer's attention and feed-forward projections
//!   the terms of the call's adapters and their selections of the rows. Only BF16 Qwen3 of 0.6B to
//!   4B runs adapters: the sizes of 8B geometry, Qwen3's, Llama's and Mistral's, lay their query,
//!   key and value rows out head-major, and a channel-scaled projection adds no update.

use super::adapters::{self, SLOTS};
use super::{
    Batch, Cache, CacheSpec, Decoder, Group, Instance, Keep, Kind, Limits, Output, Pool, Request,
    Rows, Table,
};
use crate::{AsyncModule, Checkpoint, Function, LoadError, forward};
use furiosa_opt_models::load::{Context, Load, Named, Part};
use furiosa_opt_models::models::decoder::attention::qwen3::{fp8, launch};
use furiosa_opt_models::models::decoder::attention::{Chunks, Head, KvHeads};
use furiosa_opt_models::models::decoder::attention::{Geometry, llama, qwen3, qwen3_tts};
use furiosa_opt_models::models::decoder::form;
use furiosa_opt_models::models::vision_language::qwen3_asr;
use furiosa_opt_models::modules::attention::key_value;
use furiosa_opt_models::modules::language::{Backbone, Input, Selected};
use furiosa_opt_models::modules::lora::{self, Adapters, Rank};
use furiosa_opt_models::ops::attention::full::paged::{Tiling, chunks::Masks};
use furiosa_opt_models::ops::attention::gqa::Cache as Window;
use furiosa_opt_models::ops::embedding::Ids;
use furiosa_opt_models::ops::index::copy as ops_copy;
use furiosa_opt_models::ops::index::gather::{self, Gather};
use furiosa_opt_models::ops::index::table::BlockIndex;
use furiosa_opt_models::ops::index::table::ByteOffset;
use furiosa_opt_models::ops::linear::Affine;
use furiosa_opt_models::ops::linear::affine::{self, Columns};
use furiosa_opt_models::specialize::attention::{Decode, WAVES};
use furiosa_opt_models::tensor::axis;
use furiosa_opt_models::tensor::axis::Sixteen;
use furiosa_opt_models::tensor::cache::{self, Block as CacheBlock, Blocks};
use furiosa_opt_models::tensor::mapping::Packets;
use furiosa_opt_models::tensor::memory::{Hbm, alias};
use furiosa_opt_models::tensor::parallel::{Chip, Parallel};
use furiosa_opt_models::{Module, Then};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::DeviceOutput;
use std::{convert::Infallible, io, marker::PhantomData};

/// Block IDs of a call of `R` rows whose every row is a request spanning the whole context of `W`
/// windows.
#[derive(Clone, Debug)]
struct Tables<R, W>(PhantomData<fn() -> (R, W)>);

impl<R: AxisName, W: AxisName> AxisName for Tables<R, W> {
    const NAME: Ident = Ident::new("Tables");
    const SIZE: usize = R::SIZE * W::SIZE * BlockIndex::SIZE;
}

/// What a call of the size `S` holds: the compiled context, the size's rows, one paged group of
/// every layer's blocks, and `adapters` adapters.
fn limits<S: Geometry + Decode>(adapters: usize) -> Limits {
    Limits {
        window: S::Windows::SIZE * Window::SIZE,
        rows: S::Rows::SIZE,
        cache: CacheSpec {
            pools: vec![Pool {
                block: CacheBlock::SIZE,
                bytes: Blocks::<KvHeads, Head>::BLOCK_STRIDE as usize,
            }],
            groups: vec![Group {
                pool: 0,
                layers: 0..S::LAYERS,
                kind: Kind::Paged,
                window: None,
            }],
        },
        adapters,
        overrides: 0,
    }
}

/// Each layer's blocks, viewed in the storage of a cache allocated for the decoder's spec.
fn views<L: TryFrom<Vec<Blocks<KvHeads, Head>>>>(storage: &[HbmBuffer]) -> io::Result<L> {
    let views = (storage.iter())
        .map(|layer| Blocks::from_buffer(layer.alias(..).map_err(io::Error::other)?))
        .collect::<io::Result<Vec<_>>>()?;
    Ok(views
        .try_into()
        .unwrap_or_else(|_| unreachable!("a cache of the decoder's spec holds each layer")))
}

/// The rows `kept` of a call's hidden rows, in order: a view of one ascending run, else a copy.
fn keep<W: AxisName>(
    device: &mut Device,
    hidden: &HbmBuffer,
    kept: &[usize],
) -> io::Result<Rows<bf16, m![W]>> {
    let first = kept.first().copied().unwrap_or(0);
    if (kept.iter().enumerate()).all(|(at, &row)| row == first + at) {
        return Rows::new(hidden, first..first + kept.len());
    }
    let bytes = Rows::<bf16, m![W]>::bytes();
    let target = HbmBuffer::alloc(device, kept.len() * bytes).map_err(io::Error::other)?;
    let row = |buffer: &HbmBuffer, at: usize| {
        (buffer.alias(at * bytes..(at + 1) * bytes)).map_err(io::Error::other)
    };
    let mut pairs = (kept.iter().enumerate())
        .map(|(at, &source)| Ok((row(hidden, source)?, row(&target, at)?)))
        .collect::<io::Result<Vec<_>>>()?;
    let pairs = pairs.iter_mut().map(|(from, to)| (&*from, to));
    furiosa_opt_models::ops::index::copy(&mut *device, pairs)?;
    Rows::new(&target, 0..kept.len())
}

/// The kept rows' greedy tokens of a Qwen3 head over `kept`, the rows of the size `$size`'s base
/// model.
macro_rules! greedy {
    ($size:ty, $head:expr, $device:ident, $kept:ident) => {{
        type Size = $size;
        type W = <Size as Geometry>::Hidden;
        if $kept.len() == 0 {
            return Rows::new($kept.buffer(), 0..0);
        }
        let head: qwen3::LmHead<Size, &Hbm> =
            Load::load(&mut Context::new(None), $head).map_err(io::Error::other)?;
        let bytes = Rows::<bf16, Symbol<W>>::bytes();
        let views: gather::RowStorage<W, bf16> = (0..$kept.len())
            .map(|row| alias($kept.buffer(), row * bytes..(row + 1) * bytes))
            .collect::<io::Result<_>>()?;
        let mut rows =
            gather::Rows::<bf16, W, <Size as Decode>::Rows>::new($device, &views).await?;
        let head = Function::new(&head, forward);
        let tokens = head.forward($device, (), &mut (), &mut rows).await?;
        Rows::new(&tokens.into_buffer(), 0..$kept.len())
    }};
}

/// The FP32 scores of the kept rows `$kept`, of `$hidden` features, by the head `$head` onto
/// `$vocabulary` with its bias: the rows in a block of the projection's rows, projected unrounded.
macro_rules! scored {
    ($hidden:ty, $vocabulary:ty, $head:expr, $device:ident, $kept:ident) => {{
        type W = $hidden;
        type V = $vocabulary;
        let other = io::Error::other;
        let count = $kept.len();
        let row = Rows::<bf16, Symbol<W>>::bytes();
        if count == 0 {
            let empty = HbmBuffer::alloc($device, Rows::<f32, m![V]>::bytes()).map_err(other)?;
            return Rows::new(&empty, 0..0);
        }
        let block = HbmBuffer::alloc($device, Block::SIZE * row).map_err(other)?;
        let mut target = block.alias(..count * row).map_err(other)?;
        let source = $kept.buffer().alias(..count * row).map_err(other)?;
        ops_copy($device, [(&source, &mut target)])?;
        let rows: HbmTensor<bf16, Chip, m![Block, W]> = alias(&block, ..)?;
        let head = $head;
        let projection = furiosa_opt_models::ops::linear::Unrounded::<W, V, Chip, _, _>(
            Affine::new(&head.weight, &head.bias),
            PhantomData,
        );
        let scores: HbmTensor<f32, Chip, m![Block, V]> = Function::new(&projection, forward)
            .forward($device, (), &mut (), &rows)
            .await?;
        Rows::new(&scores.into_buffer(), 0..count)
    }};
}

/// The decoders of the listed sizes `$size` whose projections hold form `$form`: the base model
/// and the causal model, which run `SLOTS` adapters a call when `adapted`, and none when their
/// front is `normalized`, Qwen3's, or `projected`, Llama's.
macro_rules! decoders {
    ($(($size:ty, $form:ty) => $kind:ident),+ $(,)?) => {$(
        /// The kept rows' hidden values.
        impl Decoder for qwen3::Model<$size, $form, Parallel<1>, cache::Bf16, Hbm> {
            const OUTPUT: Output = Output::Hidden;
            const ADAPTERS: usize = decoders!(@slots $kind);
            type Output = Rows<bf16, Symbol<<$size as Geometry>::Hidden>>;
            type Layers = <$size as Geometry>::Layers<Blocks<KvHeads, Head>>;
            type Adapter = decoders!(@adapter $kind $size);

            fn limits(&self) -> Limits {
                limits::<$size>(Self::ADAPTERS)
            }

            fn layers(storage: &[HbmBuffer]) -> io::Result<Self::Layers> {
                views(storage)
            }

            decoders!(@load $kind $size);

            async fn call(
                &self,
                device: &mut Device,
                layers: &mut Self::Layers,
                batch: Batch<'_, Self::Adapter>,
                tiling: Tiling,
            ) -> io::Result<Self::Output> {
                type Size = $size;
                type R = <Size as Decode>::Rows;
                type Work = <Size as Decode>::Work;
                let (segments, ids) = batch.row_segments(0);
                let mut reads = Work::sequential::<R>(&segments, ids, tiling)?;
                if let Some(binding) = batch.adapters() {
                    reads.extend(binding.selections::<R, Rank<3>>());
                    reads.extend(binding.selections::<R, Rank<1>>());
                }
                let (staged, views) = batch
                    .stage::<R, Tables<R, <Size as Decode>::Windows>, Chip>(
                        device,
                        &reads,
                        (0, Table::Blocks),
                    )
                    .await?;
                // Placed each call: constants the calls could share, two small uploads a call.
                let masks = Masks::new(device).await?;
                let mut views = views.into_iter();
                let mut next =
                    || views.next().ok_or_else(|| io::Error::other("a read array is missing"));
                let work = Work::bind(&mut next, &masks)?;
                let tokens: Ids<R> = alias(&staged.tokens, ..)?;
                let mut call = Input::new(tokens, staged.count, ((), (staged.history, &work)));
                let model: qwen3::Model<Size, $form, Parallel<1>, cache::Bf16, &Hbm> =
                    Load::load(&mut Context::new(None), self).map_err(io::Error::other)?;
                type W = <Size as Geometry>::Hidden;
                let rows: HbmTensor<bf16, Chip, m![R, W]> =
                    DeviceOutput::alloc(device).map_err(io::Error::other)?;
                let body = Function::new(&model, forward);
                let rows = decoders!(@run $kind, batch, next, layers, rows, body, device, call);
                keep(device, &rows.into_buffer(), batch.kept())
            }
        }

        /// The kept rows' greedy tokens.
        impl Decoder
            for qwen3::ForCausalLM<$size, $form, Parallel<1>, cache::Bf16, Hbm>
        {
            const OUTPUT: Output = Output::Tokens;
            const ADAPTERS: usize = decoders!(@slots $kind);
            type Output = Rows<i32, m![1]>;
            type Layers = <$size as Geometry>::Layers<Blocks<KvHeads, Head>>;
            type Adapter = decoders!(@adapter $kind $size);

            fn limits(&self) -> Limits {
                self.model.limits()
            }

            fn layers(storage: &[HbmBuffer]) -> io::Result<Self::Layers> {
                views(storage)
            }

            decoders!(@load $kind $size);

            async fn call(
                &self,
                device: &mut Device,
                layers: &mut Self::Layers,
                batch: Batch<'_, Self::Adapter>,
                tiling: Tiling,
            ) -> io::Result<Rows<i32, m![1]>> {
                let kept = self.model.call(device, layers, batch, tiling).await?;
                greedy!($size, &self.lm_head, device, kept)
            }
        }
    )+};
    (@slots adapted) => { SLOTS };
    (@slots $kind:ident) => { 0 };
    (@adapter adapted $size:ty) => { qwen3::Adapter<$size, Hbm> };
    (@adapter $kind:ident $size:ty) => { Infallible };
    // A PEFT checkpoint's terms are built on host from the checkpoint read once, then uploaded.
    (@load adapted $size:ty) => {
        fn adapter(
            checkpoint: &Checkpoint,
            device: &mut Device,
        ) -> Result<Self::Adapter, LoadError> {
            lora::load::<qwen3::Adapter<$size, Hbm>, qwen3::Adapter<$size, lora::Staged>>(
                checkpoint, device,
            )
        }
    };
    (@load $kind:ident $size:ty) => {};
    // A layer's front slots around its query, key and value projection's `$qkv`: Qwen3's normalize
    // each query and key head after it, Llama's take the heads as projected.
    (@front projected $qkv:expr) => { $qkv };
    (@front $kind:ident $qkv:expr) => { ($qkv, (), ()) };
    // Each layer's slots: its front's, its cache blocks and its output projection's, then its
    // feed-forward's gate, up and down projections'.
    (@run adapted, $batch:ident, $next:ident, $layers:ident, $rows:ident, $body:ident, $device:ident, $call:ident) => {{
        match $batch.adapters() {
            None => decoders!(@run normalized, $batch, $next, $layers, $rows, $body, $device, $call),
            Some(binding) => {
                let packed = adapters::selected::<R, Rank<3>>(&mut $next)?;
                let output = adapters::selected::<R, Rank<1>>(&mut $next)?;
                let held = binding.held();
                let mut layer = 0;
                let slots = $layers.each_mut().map(|blocks| {
                    let terms = held.map(|adapter| &adapter.0.0.0.0[layer]);
                    layer += 1;
                    let (attention, mlp) = (terms.map(|t| &t.0.0), terms.map(|t| &t.1.0));
                    let qkv = Adapters {
                        terms: attention.map(|terms| terms.0.term()),
                        selections: &packed,
                    };
                    let o = Adapters {
                        terms: attention.map(|terms| &terms.1.0),
                        selections: &output,
                    };
                    let gate = Adapters {
                        terms: mlp.map(|terms| &terms.0.0),
                        selections: &output,
                    };
                    let up = Adapters {
                        terms: mlp.map(|terms| &terms.1.0),
                        selections: &output,
                    };
                    let down = Adapters {
                        terms: mlp.map(|terms| &terms.2.0),
                        selections: &output,
                    };
                    ((((qkv, ()), (), ()), blocks, o), (((gate, ()), up), ((), down)))
                });
                let mut state = ((((), slots), ()), $rows);
                $body.forward($device, (), &mut state, &mut $call).await?;
                state.1
            }
        }
    }};
    (@run $kind:ident, $batch:ident, $next:ident, $layers:ident, $rows:ident, $body:ident, $device:ident, $call:ident) => {{
        let slots = $layers.each_mut().map(|blocks| {
            ((decoders!(@front $kind ((), ())), blocks, ()), ((((), ()), ()), ((), ())))
        });
        let mut state = ((((), slots), ()), $rows);
        $body.forward($device, (), &mut state, &mut $call).await?;
        state.1
    }};
}

decoders!(
    (qwen3::Size<"0.6B">, form::Affine) => adapted,
    (qwen3::Size<"1.7B">, form::Affine) => adapted,
    (qwen3::Size<"4B">, form::Affine) => adapted,
    (qwen3::Size<"8B">, form::Affine) => normalized,
    (qwen3_tts::Talker<"1.7B">, form::Affine) => normalized,
    (qwen3_tts::Predictor<"1.7B">, form::Affine) => normalized,
    (llama::Size<"8B">, form::Affine) => projected,
    (llama::Size<"Mistral-7B">, form::Affine) => projected,
    (qwen3::Size<"0.6B">, form::ChannelScaled) => normalized,
    (qwen3::Size<"1.7B">, form::ChannelScaled) => normalized,
    (qwen3::Size<"4B">, form::ChannelScaled) => normalized,
);

/// A conditional decoder whose calls take override rows: the decoder, and a second owner of its
/// token table's storage, whose rows past the checkpoint's tokens each call's override rows fill.
pub struct Overriding<D> {
    decoder: D,
    table: HbmBuffer,
}

/// The decoders of the listed conditional checkpoints `$model` of the Qwen3 geometry `$size`: the
/// base model `$body`'s call, its rows overridden by the caller's, then the greedy head `$head`.
/// A call's override rows fill the table's last rows, one a call row, so a checkpoint's tokens end
/// before them.
macro_rules! conditionals {
    ($($model:ty, $size:ty, $body:ident, $head:ident);+ $(;)?) => {$(
        /// The model reads a whole checkpoint and serves its calls with a second owner of its
        /// token table's storage.
        impl crate::registry::Listed for $model {
            fn reads(part: Part) -> Result<(), LoadError> {
                match part {
                    Part::Whole => Ok(()),
                    Part::Encoder => Err(LoadError::Unsupported {
                        name: "encoder".into(),
                        reason: "a decoder holds no media encoder that loads alone".into(),
                    }),
                }
            }

            fn served(mut self, tiling: Tiling) -> crate::registry::Model {
                let Then(Then(Backbone { embedding: Named(table), body }, norm), store) =
                    self.$body.body;
                let (table, storage) = (table.shared())
                    .unwrap_or_else(|_| unreachable!("a buffer aliases itself"));
                self.$body.body =
                    Then(Then(Backbone { embedding: Named(table), body }, norm), store);
                let decoder = Overriding { decoder: self, table: storage };
                crate::registry::Model::Decoder(Box::new(Instance::new(decoder, tiling)))
            }
        }

        /// The kept rows' greedy tokens.
        impl Decoder for Overriding<$model> {
            const OUTPUT: Output = Output::Tokens;
            type Output = Rows<i32, m![1]>;
            type Layers = <$size as Geometry>::Layers<Blocks<KvHeads, Head>>;
            /// The base model's, of which the conditional model's calls run none.
            type Adapter = <Base<$size> as Decoder>::Adapter;

            fn limits(&self) -> Limits {
                Limits {
                    adapters: <Self as Decoder>::ADAPTERS,
                    overrides: <$size as Geometry>::Hidden::SIZE,
                    ..self.decoder.$body.limits()
                }
            }

            fn layers(storage: &[HbmBuffer]) -> io::Result<Self::Layers> {
                views(storage)
            }

            async fn call(
                &self,
                device: &mut Device,
                layers: &mut Self::Layers,
                batch: Batch<'_, Self::Adapter>,
                tiling: Tiling,
            ) -> io::Result<Rows<i32, m![1]>> {
                type W = <$size as Geometry>::Hidden;
                let first = <$size as Geometry>::Vocabulary::SIZE - <$size as Decode>::Rows::SIZE;
                let row = Rows::<bf16, Symbol<W>>::bytes();
                let batch = batch.place(device, &self.table, (first, row)).await?;
                let kept = self.decoder.$body.call(device, layers, batch, tiling).await?;
                greedy!($size, &self.decoder.$head, device, kept)
            }
        }
    )+};
}

conditionals!(
    qwen3_asr::ForConditionalGeneration<
        qwen3::Size<"1.7B">,
        form::Affine,
        Parallel<1>,
        cache::Bf16,
        Hbm,
    >,
    qwen3::Size<"1.7B">,
    language_model,
    lm_head;
    qwen3_tts::ForConditionalGeneration<
        qwen3_tts::Talker<"1.7B">,
        form::Affine,
        Parallel<1>,
        cache::Bf16,
        Hbm,
    >,
    qwen3_tts::Talker<"1.7B">,
    model,
    codec_head;
);

/// Qwen3-TTS 1.7B's BF16 code predictor on one chip.
pub type CodePredictor = qwen3_tts::CodePredictor<
    qwen3_tts::Predictor<"1.7B">,
    form::Affine,
    Parallel<1>,
    cache::Bf16,
    Hbm,
>;

type Predictor = qwen3_tts::Predictor<"1.7B">;

/// The rows a predictor call projects at once: a block of the projection's rows, past its rows.
type Block = axis::Rows<128>;

/// The predictor reads a whole checkpoint and serves its calls with a second owner of its body's
/// token table's storage, whose rows from the codebook's entries on take each call's input rows.
impl crate::registry::Listed for CodePredictor {
    fn reads(part: Part) -> Result<(), LoadError> {
        match part {
            Part::Whole => Ok(()),
            Part::Encoder => Err(LoadError::Unsupported {
                name: "encoder".into(),
                reason: "a decoder holds no media encoder that loads alone".into(),
            }),
        }
    }

    fn served(mut self, tiling: Tiling) -> crate::registry::Model {
        let Then(
            Then(
                Backbone {
                    embedding: Named(table),
                    body,
                },
                norm,
            ),
            store,
        ) = self.model.body;
        let (table, storage) =
            (table.shared()).unwrap_or_else(|_| unreachable!("a buffer aliases itself"));
        self.model.body = Then(
            Then(
                Backbone {
                    embedding: Named(table),
                    body,
                },
                norm,
            ),
            store,
        );
        let decoder = Overriding {
            decoder: self,
            table: storage,
        };
        crate::registry::Model::Decoder(Box::new(Instance::new(decoder, tiling)))
    }
}

/// Each kept row's pick of the entry its position names, every kept row at one position: row
/// `q > 0` of a frame's sequence picks entry `q` by head `q - 1`. A call's row is the override
/// it gives, the talker's hidden row at position 0, or its token's row of the table its position
/// names; the call projects them into the body's table, from row 2048 on, which its token IDs
/// then name.
impl Decoder for Overriding<CodePredictor> {
    const OUTPUT: Output = Output::Tokens;
    type Output = Rows<i32, m![1]>;
    type Layers = <Predictor as Geometry>::Layers<Blocks<KvHeads, Head>>;
    /// The base model's, of which the predictor's calls run none.
    type Adapter = <Base<Predictor> as Decoder>::Adapter;

    fn limits(&self) -> Limits {
        Limits {
            adapters: <Self as Decoder>::ADAPTERS,
            overrides: qwen3_tts::Text::SIZE,
            ..self.decoder.model.limits()
        }
    }

    fn layers(storage: &[HbmBuffer]) -> io::Result<Self::Layers> {
        views(storage)
    }

    async fn call(
        &self,
        device: &mut Device,
        layers: &mut Self::Layers,
        batch: Batch<'_, Self::Adapter>,
        tiling: Tiling,
    ) -> io::Result<Rows<i32, m![1]>> {
        let (step, kept) = self.predict(device, layers, batch, tiling).await?;
        greedy!(Predictor, &self.decoder.lm_head[step - 1], device, kept)
    }
}

impl Overriding<CodePredictor> {
    /// The kept rows' hidden values and the position they share, whose entry they pick.
    async fn predict(
        &self,
        device: &mut Device,
        layers: &mut <Self as Decoder>::Layers,
        batch: Batch<'_, <Self as Decoder>::Adapter>,
        tiling: Tiling,
    ) -> io::Result<(usize, Rows<bf16, Symbol<<Predictor as Geometry>::Hidden>>)> {
        type Text = qwen3_tts::Text;
        type W = <Predictor as Geometry>::Hidden;
        let unsupported = |message: &str| io::Error::new(io::ErrorKind::Unsupported, message);
        let invalid = |message: &str| io::Error::new(io::ErrorKind::InvalidInput, message);
        let other = io::Error::other;
        let positions = batch.positions();
        let mut steps = batch.kept().iter().map(|&row| positions[row]);
        let step = steps.next().unwrap_or(1);
        if steps.any(|other| other != step) {
            return Err(unsupported(
                "the kept rows of a predictor call share one position",
            ));
        }
        if !(1..=qwen3_tts::GROUPS).contains(&step) {
            return Err(invalid(
                "a kept row picks an entry of a later codebook, at 1 to 15",
            ));
        }
        if positions
            .iter()
            .any(|&position| position > qwen3_tts::GROUPS)
        {
            return Err(invalid("a frame's sequence holds 16 rows"));
        }
        // Each call row's input, of the talker's width, in a block of the projection's rows.
        let text = Rows::<bf16, Symbol<Text>>::bytes();
        let inputs = HbmBuffer::alloc(device, Block::SIZE * text).map_err(other)?;
        let row = |buffer: &HbmBuffer, at: usize| {
            (buffer.alias(at * text..(at + 1) * text)).map_err(io::Error::other)
        };
        let mut given = vec![false; positions.len()];
        let mut pairs = Vec::new();
        for (rows, values) in batch.overrides() {
            for (at, call) in rows.enumerate() {
                given[call] = true;
                pairs.push((row(values, at)?, row(&inputs, call)?));
            }
        }
        if (positions.iter().zip(&given)).any(|(&position, &given)| position == 0 && !given) {
            return Err(invalid(
                "a frame's first row is the talker's hidden row, an override",
            ));
        }
        // The rows of each table, gathered by their tokens' byte offsets.
        let lanes = <axis::Lanes<Block> as AxisName>::SIZE;
        let word = Rows::<i32, m![1]>::bytes();
        let ids_bytes = HbmTensor::<i32, Chip, Packets<Block>>::size();
        for table in 1..=qwen3_tts::GROUPS {
            let rows: Vec<usize> = (0..positions.len())
                .filter(|&call| !given[call] && positions[call] == table)
                .collect();
            if rows.is_empty() {
                continue;
            }
            let zeros = vec![0; ids_bytes];
            let ids = HostBufferSlice::from(&zeros[..])
                .to_hbm(&mut device.pdma)
                .await
                .map_err(other)?;
            let mut tokens = (rows.iter().enumerate())
                .map(|(at, &call)| {
                    let at = (at / lanes) * 8 + at % lanes;
                    Ok((
                        batch.token(call)?,
                        ids.alias(at * word..(at + 1) * word).map_err(other)?,
                    ))
                })
                .collect::<io::Result<Vec<_>>>()?;
            ops_copy(device, tokens.iter_mut().map(|(from, to)| (&*from, to)))?;
            let ids: HbmTensor<i32, Chip, Packets<Block>> = alias(&ids, ..)?;
            let offsets: HbmTensor<i32, Chip, m![Block]> =
                Function::new(&ByteOffset::<bf16, m![Text], m![Block]>::default(), forward)
                    .forward(device, (), &mut (), &ids)
                    .await?;
            let gathered: HbmTensor<bf16, Chip, m![Block, Text]> = match table {
                1 => {
                    Function::new(&Gather, forward)
                        .forward(device, (), &mut (), (&self.decoder.talker_table, &offsets))
                        .await?
                }
                _ => {
                    let source = &self.decoder.tables[table - 2];
                    Function::new(&Gather, forward)
                        .forward(device, (), &mut (), (source, &offsets))
                        .await?
                }
            };
            let gathered = gathered.into_buffer();
            for (at, &call) in rows.iter().enumerate() {
                pairs.push((row(&gathered, at)?, row(&inputs, call)?));
            }
        }
        ops_copy(device, pairs.iter_mut().map(|(from, to)| (&*from, to)))?;
        // Projected into the body's width, into the body's table from the codebook's entries on.
        let inputs: HbmTensor<bf16, Chip, m![Block, Text]> = alias(&inputs, ..)?;
        let projection = &self.decoder.small_to_mtp_projection;
        let projection = Affine::new(&projection.weight, &projection.bias);
        let projected: HbmTensor<bf16, Chip, m![Block, W]> = Function::new(&projection, forward)
            .forward(device, (), &mut (), &inputs)
            .await?;
        // The call's rows alone, which its token IDs name: a whole block's copy costs as much as
        // the predictor's layers.
        let first = qwen3_tts::PROJECTED;
        let width = Rows::<bf16, Symbol<W>>::bytes();
        let live = positions.len();
        let mut placed =
            (self.table.alias(first * width..(first + live) * width)).map_err(other)?;
        let projected = projected.into_buffer();
        let projected = projected.alias(..live * width).map_err(other)?;
        ops_copy(device, [(&projected, &mut placed)])?;
        let batch = batch.named(device, first).await?;
        let kept = self
            .decoder
            .model
            .call(device, layers, batch, tiling)
            .await?;
        Ok((step, kept))
    }
}

/// The code predictor whose calls return each kept row's FP32 scores of the entry its position
/// names, the padded rows past the codebook's lowest, for a caller that samples them.
///
/// - Every entry's row of the talker's table and of the predictor's tables is projected into the
///   body's token table once, at the first call ([`Scoring::row`]); the rows a call gives are
///   projected into rows of their own past them ([`Scoring::given`]). A call row's token ID names
///   its row, so a call launches the body and the head alone.
pub struct Scoring {
    predictor: Overriding<CodePredictor>,
    /// Second owners of the talker's table and of the tables of positions 2 to 15, which the
    /// projected rows are computed from.
    sources: Vec<HbmBuffer>,
    /// Set once the token table holds every source row projected.
    projected: std::sync::OnceLock<()>,
}

type Talked = qwen3_tts::Talked;

/// `tensor`, and a second owner of its storage.
fn second_owner<E: M>(tensor: HbmTensor<bf16, Chip, E>) -> (HbmTensor<bf16, Chip, E>, HbmBuffer) {
    let buffer = tensor.into_buffer();
    let second = (buffer.alias(..)).unwrap_or_else(|_| unreachable!("a buffer aliases itself"));
    let tensor = alias(&buffer, ..).unwrap_or_else(|_| unreachable!("a buffer is its own tensor"));
    (tensor, second)
}

type PredictorHidden = <Predictor as Geometry>::Hidden;
type PredictorVocabulary = <Predictor as Geometry>::Vocabulary;

impl Decoder for Scoring {
    const OUTPUT: Output = Output::Scores;
    type Output = Rows<f32, m![PredictorVocabulary]>;
    type Layers = <Overriding<CodePredictor> as Decoder>::Layers;
    type Adapter = <Overriding<CodePredictor> as Decoder>::Adapter;

    fn limits(&self) -> Limits {
        Decoder::limits(&self.predictor)
    }

    fn layers(storage: &[HbmBuffer]) -> io::Result<Self::Layers> {
        views(storage)
    }

    async fn call(
        &self,
        device: &mut Device,
        layers: &mut Self::Layers,
        batch: Batch<'_, Self::Adapter>,
        tiling: Tiling,
    ) -> io::Result<Rows<f32, m![PredictorVocabulary]>> {
        let (step, kept) = self.predict(device, layers, batch, tiling).await?;
        let head = &self.predictor.decoder.lm_head[step - 1].head;
        scored!(PredictorHidden, PredictorVocabulary, head, device, kept)
    }
}

impl Scoring {
    /// The predictor of `sources`' tables, which its first call projects into its token table.
    fn new(predictor: Overriding<CodePredictor>, sources: Vec<HbmBuffer>) -> Self {
        Self {
            predictor,
            sources,
            projected: std::sync::OnceLock::new(),
        }
    }

    /// The first row of the token table that holds the projected entries a row at `position`
    /// reads, and their count.
    fn span(position: usize) -> (usize, usize) {
        match position {
            1 => (0, Talked::SIZE),
            _ => (
                Talked::SIZE + (position - 2) * qwen3_tts::Codebook::SIZE,
                qwen3_tts::Codebook::SIZE,
            ),
        }
    }

    /// The token-table row of `entry`, of the table a row at `position` reads: the token ID of
    /// that row.
    pub fn row(position: usize, entry: usize) -> usize {
        Self::span(position).0 + entry
    }

    /// The token-table row that a call's own row `call` is placed at: the token ID of that row.
    pub const fn given(call: usize) -> usize {
        qwen3_tts::PROJECTED + call
    }

    /// Every source row projected into the body's width, into the token table, a block of rows a
    /// launch, as a call projects its own rows; at the first call.
    fn projected(&self, device: &mut Device) -> io::Result<()> {
        if self.projected.get().is_some() {
            return Ok(());
        }
        type Text = qwen3_tts::Text;
        type W = <Predictor as Geometry>::Hidden;
        let text = Rows::<bf16, Symbol<Text>>::bytes();
        let width = Rows::<bf16, Symbol<W>>::bytes();
        let projection = &self.predictor.decoder.small_to_mtp_projection;
        let projection = Affine::new(&projection.weight, &projection.bias);
        let function = Function::new(&projection, forward);
        for (position, source) in (1..=qwen3_tts::GROUPS).zip(&self.sources) {
            let (start, rows) = Self::span(position);
            for block in (0..rows).step_by(Block::SIZE) {
                let input: HbmTensor<bf16, Chip, m![Block, Text]> =
                    alias(source, block * text..(block + Block::SIZE) * text)?;
                let at = start + block;
                let mut output: HbmTensor<bf16, Chip, m![Block, W]> = alias(
                    &self.predictor.table,
                    at * width..(at + Block::SIZE) * width,
                )?;
                function.forward_into(device, (), &mut (), &input, &mut output)?;
            }
        }
        self.projected.get_or_init(|| ());
        Ok(())
    }

    /// The kept rows' hidden values and the position they share, whose entry they pick: each row's
    /// token names its row of the token table, the rows the call gives projected into theirs.
    async fn predict(
        &self,
        device: &mut Device,
        layers: &mut <Self as Decoder>::Layers,
        batch: Batch<'_, <Self as Decoder>::Adapter>,
        tiling: Tiling,
    ) -> io::Result<(usize, Rows<bf16, Symbol<PredictorHidden>>)> {
        type Text = qwen3_tts::Text;
        type W = PredictorHidden;
        let unsupported = |message: &str| io::Error::new(io::ErrorKind::Unsupported, message);
        let invalid = |message: &str| io::Error::new(io::ErrorKind::InvalidInput, message);
        let other = io::Error::other;
        let positions = batch.positions();
        let mut steps = batch.kept().iter().map(|&row| positions[row]);
        let step = steps.next().unwrap_or(1);
        if steps.any(|other| other != step) {
            return Err(unsupported(
                "the kept rows of a predictor call share one position",
            ));
        }
        if !(1..=qwen3_tts::GROUPS).contains(&step) {
            return Err(invalid(
                "a kept row picks an entry of a later codebook, at 1 to 15",
            ));
        }
        if positions
            .iter()
            .any(|&position| position > qwen3_tts::GROUPS)
        {
            return Err(invalid("a frame's sequence holds 16 rows"));
        }
        self.projected(device)?;
        let (text, width) = (
            Rows::<bf16, Symbol<Text>>::bytes(),
            Rows::<bf16, Symbol<W>>::bytes(),
        );
        let live = positions.len();
        let mut given = vec![false; live];
        // The rows the call gives, the talker's hidden rows, projected into their own rows.
        let overrides: Vec<_> = batch.overrides().collect();
        if !overrides.is_empty() {
            let inputs = HbmBuffer::alloc(device, Block::SIZE * text).map_err(other)?;
            let mut rows = Vec::new();
            for (calls, values) in overrides {
                for (at, call) in calls.enumerate() {
                    given[call] = true;
                    let value = (values.alias(at * text..(at + 1) * text)).map_err(other)?;
                    let input = (inputs.alias(call * text..(call + 1) * text)).map_err(other)?;
                    rows.push((value, input));
                }
            }
            ops_copy(device, rows.iter_mut().map(|(from, to)| (&*from, to)))?;
            let inputs: HbmTensor<bf16, Chip, m![Block, Text]> = alias(&inputs, ..)?;
            let projection = &self.predictor.decoder.small_to_mtp_projection;
            let projection = Affine::new(&projection.weight, &projection.bias);
            let projected: HbmTensor<bf16, Chip, m![Block, W]> =
                Function::new(&projection, forward)
                    .forward(device, (), &mut (), &inputs)
                    .await?;
            let projected = projected.into_buffer();
            let mut pairs = (0..live)
                .filter(|&call| given[call])
                .map(|call| {
                    let row = Self::given(call);
                    Ok((
                        (projected.alias(call * width..(call + 1) * width)).map_err(other)?,
                        (self.predictor.table.alias(row * width..(row + 1) * width))
                            .map_err(other)?,
                    ))
                })
                .collect::<io::Result<Vec<_>>>()?;
            ops_copy(device, pairs.iter_mut().map(|(from, to)| (&*from, to)))?;
        }
        if (positions.iter().zip(&given)).any(|(&position, &given)| position == 0 && !given) {
            return Err(invalid(
                "a frame's first row is the talker's hidden row, an override",
            ));
        }
        let kept = (self.predictor.decoder.model)
            .call(device, layers, batch, tiling)
            .await?;
        Ok((step, kept))
    }
}

/// The token table of a base model's body `$body` with a second owner of its storage: the body
/// placed back, and the storage.
macro_rules! shared {
    ($body:expr) => {{
        let Then(
            Then(
                Backbone {
                    embedding: Named(table),
                    body,
                },
                norm,
            ),
            store,
        ) = $body;
        let (table, storage) =
            (table.shared()).unwrap_or_else(|_| unreachable!("a buffer aliases itself"));
        (
            Then(
                Then(
                    Backbone {
                        embedding: Named(table),
                        body,
                    },
                    norm,
                ),
                store,
            ),
            storage,
        )
    }};
}

/// Qwen3-TTS 1.7B's BF16 talker on one chip.
type TalkerModel = qwen3_tts::ForConditionalGeneration<
    qwen3_tts::Talker<"1.7B">,
    form::Affine,
    Parallel<1>,
    cache::Bf16,
    Hbm,
>;

type Talker = qwen3_tts::Talker<"1.7B">;

/// Qwen3-TTS 1.7B's talker and code predictor on one chip, served a frame of entries a talker row.
pub type Speaker = qwen3_tts::Speaker<TalkerModel, CodePredictor>;

/// The speaker reads a whole checkpoint; each of its models serves with a second owner of its
/// body's token table.
impl crate::registry::Listed for Speaker {
    fn reads(part: Part) -> Result<(), LoadError> {
        match part {
            Part::Whole => Ok(()),
            Part::Encoder => Err(LoadError::Unsupported {
                name: "encoder".into(),
                reason: "a decoder holds no media encoder that loads alone".into(),
            }),
        }
    }

    fn served(mut self, tiling: Tiling) -> crate::registry::Model {
        let (body, table) = shared!(self.talker.model.body);
        self.talker.model.body = body;
        let talker = Overriding {
            decoder: self.talker,
            table,
        };
        let (body, table) = shared!(self.predictor.model.body);
        self.predictor.model.body = body;
        // The tables of positions 1 to 15, whose rows the predictor projects once.
        let (talker_table, talked) = second_owner(self.predictor.talker_table);
        self.predictor.talker_table = talker_table;
        let (tables, mut sources): (Vec<_>, Vec<_>) =
            self.predictor.tables.map(second_owner).into_iter().unzip();
        self.predictor.tables = (tables.try_into())
            .unwrap_or_else(|_| unreachable!("as many tables as the predictor's"));
        sources.insert(0, talked);
        sources.truncate(qwen3_tts::GROUPS);
        let predictor = Scoring::new(
            Overriding {
                decoder: self.predictor,
                table,
            },
            sources,
        );
        let speaking = Speaking {
            talker,
            predictor: Instance::new(predictor, tiling),
            embeddings: self.embeddings,
            last: std::sync::Mutex::new(None),
        };
        crate::registry::Model::Decoder(Box::new(Instance::new(speaking, tiling)))
    }
}

/// The speaker's rows' axes: the talker's width, and its embeddings' tables and projections.
type Spoken = qwen3_tts::Text;
type TtsStacked = qwen3_tts::Stacked;
type TtsTextWidth = qwen3_tts::TextWidth;
type TtsTextInner = qwen3_tts::TextInner;
type TtsTextVocabulary = qwen3_tts::TextVocabulary;
type TtsCodecRows = qwen3_tts::CodecRows;

/// The speaker as a decoder: a call runs the talker over its requests' rows, every row an override,
/// and returns each kept row's FP32 scores of its frame's first entry, the talker head's, its
/// suppressed control rows lowest.
///
/// - `frames` then gives the frames of the call's kept rows whose first entries the caller drew:
///   the code predictor, fed each row's hidden values on a cache of its own, scores each later
///   entry, which the host draws as a [`Sampling`] says.
/// - `rows` gives the rows a frame and a text row give the talker next; `text`, `codec` and `add`
///   a prompt's rows; all on the device.
pub struct Speaking {
    talker: Overriding<TalkerModel>,
    predictor: Instance<Scoring>,
    embeddings: qwen3_tts::Embeddings,
    /// The last call's kept hidden rows, which `frames` feeds the predictor.
    last: std::sync::Mutex<Option<Rows<bf16, Symbol<TalkerHidden>>>>,
}

type TalkerHidden = <Talker as Geometry>::Hidden;
type TalkerVocabulary = <Talker as Geometry>::Vocabulary;

/// A frame's picks: sixteen rows of the stacked tables a row of a block.
type Picks = axis::Rows<2048>;

/// Each kept row's scores of the talker's rows, FP32.
impl Decoder for Speaking {
    const OUTPUT: Output = Output::Scores;
    type Output = Rows<f32, m![TalkerVocabulary]>;
    type Layers = <Talker as Geometry>::Layers<Blocks<KvHeads, Head>>;
    /// The talker's base model's, of which the speaker's calls run none.
    type Adapter = <Base<Talker> as Decoder>::Adapter;

    fn limits(&self) -> Limits {
        Decoder::limits(&self.talker)
    }

    fn layers(storage: &[HbmBuffer]) -> io::Result<Self::Layers> {
        views(storage)
    }

    async fn call(
        &self,
        device: &mut Device,
        layers: &mut Self::Layers,
        batch: Batch<'_, Self::Adapter>,
        tiling: Tiling,
    ) -> io::Result<Rows<f32, m![TalkerVocabulary]>> {
        let first = TalkerVocabulary::SIZE - <Talker as Decode>::Rows::SIZE;
        let row = Rows::<bf16, Symbol<TalkerHidden>>::bytes();
        let batch = batch
            .place(device, &self.talker.table, (first, row))
            .await?;
        let hidden = self
            .talker
            .decoder
            .model
            .call(device, layers, batch, tiling)
            .await?;
        // The head's scores with its bias, which lowers the control rows.
        let head = &self.talker.decoder.codec_head.head;
        let scores: io::Result<Rows<f32, m![TalkerVocabulary]>> =
            async { scored!(TalkerHidden, TalkerVocabulary, head, device, hidden) }.await;
        *self
            .last
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(hidden);
        scores
    }
}

impl Speaking {
    /// The frames of the last call's kept rows whose first entries are `firsts`, in kept order:
    /// each row's sixteen entries, the other fifteen picked as `sampling` says from the code
    /// predictor's FP32 scores, the predictor fed the row's hidden values and the entries before,
    /// each row's draws from its seed of `seeds`. A call's frames are given once.
    pub async fn frames(
        &self,
        device: &mut Device,
        firsts: &[i32],
        sampling: Sampling,
        seeds: &[u64],
    ) -> io::Result<Vec<[i32; Sixteen::SIZE]>> {
        let other = io::Error::other;
        let invalid =
            |message: &str| io::Error::new(io::ErrorKind::InvalidInput, message.to_string());
        let hidden = (self
            .last
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner))
        .take()
        .ok_or_else(|| invalid("frames follow a call of the talker"))?;
        let count = hidden.len();
        if firsts.len() != count || seeds.len() != count {
            return Err(invalid("a first entry and a seed a kept row of the call"));
        }
        let mut frames: Vec<[i32; Sixteen::SIZE]> = (firsts.iter())
            .map(|&first| std::array::from_fn(|group| if group == 0 { first } else { 0 }))
            .collect();
        if count == 0 {
            return Ok(frames);
        }
        let mut draws: Vec<Draws> = seeds.iter().map(|&seed| Draws(seed)).collect();
        let row = Rows::<bf16, Symbol<TalkerHidden>>::bytes();
        let predictor = &self.predictor;
        let limits = Decoder::limits(predictor.decoder());
        // A call's token IDs one after another, then token 0 to the call's compiled rows, which the
        // call reads in place.
        let upload = async |device: &mut Device, words: &[i32]| -> io::Result<HbmBuffer> {
            let mut bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            bytes.resize(bytes.len().max(limits.rows * size_of::<i32>()), 0);
            HostBufferSlice::from(&bytes[..])
                .to_hbm(&mut device.pdma)
                .await
                .map_err(other)
        };
        let cache = Cache::new(device, &limits.cache, count)?;
        let entries = PredictorVocabulary::SIZE;
        let mut host = vec![0u8; count * entries * 4];
        for position in 1..=qwen3_tts::GROUPS {
            // A row's first call reads its hidden row, then its first entry; each later one reads
            // its entry before: each by its token-table row.
            let named = |position: usize, entry: i32| -> io::Result<i32> {
                let entry = usize::try_from(entry)
                    .ok()
                    .filter(|&entry| entry < Scoring::span(position).1)
                    .ok_or_else(|| invalid("an entry lies outside its table"))?;
                Ok(Scoring::row(position, entry) as i32)
            };
            let words: Vec<i32> = match position {
                1 => (firsts.iter().enumerate())
                    .map(|(at, &first)| Ok([Scoring::given(2 * at) as i32, named(1, first)?]))
                    .collect::<io::Result<Vec<_>>>()?
                    .concat(),
                _ => (frames.iter())
                    .map(|frame| named(position, frame[position - 1]))
                    .collect::<io::Result<_>>()?,
            };
            let tokens = upload(device, &words).await?;
            let requests = (0..count)
                .map(|at| {
                    let (cached, tokens, overrides) = match position {
                        1 => (
                            0,
                            Rows::new(&tokens, 2 * at..2 * at + 2)?,
                            vec![(
                                0,
                                hidden
                                    .buffer()
                                    .alias(at * row..(at + 1) * row)
                                    .map_err(other)?,
                            )],
                        ),
                        _ => (position, Rows::new(&tokens, at..at + 1)?, Vec::new()),
                    };
                    Ok(Request {
                        rows: vec![vec![at as u32]],
                        cached,
                        tokens,
                        keep: Keep::Last,
                        adapter: None,
                        positions: None,
                        mask: None,
                        overrides,
                    })
                })
                .collect::<io::Result<Vec<_>>>()?;
            let scores: Rows<f32, m![PredictorVocabulary]> =
                AsyncModule::forward(predictor, device, (), &cache, &requests[..]).await?;
            (scores
                .buffer()
                .to_host(&mut device.pdma)
                .output(&mut host[..])
                .await)
                .map_err(other)?;
            for (at, frame) in frames.iter_mut().enumerate() {
                let scores: Vec<f32> = (host[at * entries * 4..(at + 1) * entries * 4]
                    .as_chunks::<4>()
                    .0
                    .iter())
                .take(qwen3_tts::Codebook::SIZE)
                .map(|&bytes| f32::from_le_bytes(bytes))
                .collect();
                frame[position] = sampling.pick(&scores, draws[at].uniform()) as i32;
            }
        }
        Ok(frames)
    }

    /// The rows the talker reads after `frames`: each frame's sixteen table rows summed in FP32 and
    /// rounded once, then its text row of `texts` added, as Transformers sums them.
    pub async fn rows(
        &self,
        device: &mut Device,
        frames: &[[i32; Sixteen::SIZE]],
        texts: &[HbmBuffer],
    ) -> io::Result<Rows<bf16, m![Spoken]>> {
        let other = io::Error::other;
        let invalid =
            |message: &str| io::Error::new(io::ErrorKind::InvalidInput, message.to_string());
        let count = frames.len();
        if count > Block::SIZE || texts.len() != count {
            return Err(invalid(
                "rows of at most a block of frames, a text row each",
            ));
        }
        let row = Rows::<bf16, m![Spoken]>::bytes();
        let mut offsets = vec![0i32; Picks::SIZE];
        for (at, frame) in frames.iter().enumerate() {
            for (group, &code) in frame.iter().enumerate() {
                let limit = if group == 0 { 3072 } else { 2048 };
                let code = usize::try_from(code)
                    .ok()
                    .filter(|&code| code < limit)
                    .ok_or_else(|| invalid("an entry lies outside its codebook"))?;
                offsets[at * Sixteen::SIZE + group] =
                    ((qwen3_tts::codec_row(group) + code) * row) as i32;
            }
        }
        let bytes: Vec<u8> = offsets
            .iter()
            .flat_map(|offset| offset.to_le_bytes())
            .collect();
        let offsets: HbmTensor<i32, Chip, m![Picks]> = alias(
            &HostBufferSlice::from(&bytes[..])
                .to_hbm(&mut device.pdma)
                .await
                .map_err(other)?,
            ..,
        )?;
        let picked: HbmTensor<bf16, Chip, m![Picks, Spoken]> = Function::new(&Gather, forward)
            .forward(device, (), &mut (), (&self.embeddings.codecs, &offsets))
            .await?;
        let stacked: HbmTensor<bf16, Chip, m![Block, TtsStacked]> =
            alias(&picked.into_buffer(), ..)?;
        let summed: HbmTensor<bf16, Chip, m![Block, Spoken]> =
            Function::new(&self.embeddings.sum, forward)
                .forward(device, (), &mut (), &stacked)
                .await?;
        let text = HbmBuffer::alloc(device, Block::SIZE * row).map_err(other)?;
        let mut pairs = (texts.iter().enumerate())
            .map(|(at, source)| Ok((source, text.alias(at * row..(at + 1) * row).map_err(other)?)))
            .collect::<io::Result<Vec<_>>>()?;
        ops_copy(device, pairs.iter_mut().map(|(from, to)| (*from, to)))?;
        let text: HbmTensor<bf16, Chip, m![Block, Spoken]> = alias(&text, ..)?;
        let rows = Function::new(&furiosa_opt_models::ops::pointwise::add::Add, forward)
            .forward(device, (), &mut (), (&summed, &text))
            .await?;
        Rows::new(&rows.into_buffer(), 0..count)
    }

    /// The rows of `pairs`, each the BF16 sum of its two rows, a missing one zero, as Transformers
    /// adds a prompt's text rows and codec rows: a text row and a codec row, the speaker's
    /// embedding, or a text row alone.
    pub async fn add(
        &self,
        device: &mut Device,
        pairs: &[(Option<HbmBuffer>, Option<HbmBuffer>)],
    ) -> io::Result<Rows<bf16, m![Spoken]>> {
        let other = io::Error::other;
        let row = Rows::<bf16, m![Spoken]>::bytes();
        let output = HbmBuffer::alloc(device, pairs.len().max(1) * row).map_err(other)?;
        let zeros = vec![0u8; Block::SIZE * row];
        for (at, block) in pairs.chunks(Block::SIZE).enumerate() {
            let mut sides = Vec::with_capacity(2);
            for side in 0..2 {
                let buffer = (HostBufferSlice::from(&zeros[..])
                    .to_hbm(&mut device.pdma)
                    .await)
                    .map_err(other)?;
                let mut copies = (block.iter().enumerate())
                    .filter_map(|(at, pair)| {
                        let source = if side == 0 {
                            pair.0.as_ref()
                        } else {
                            pair.1.as_ref()
                        }?;
                        Some(
                            buffer
                                .alias(at * row..(at + 1) * row)
                                .map(|target| (source, target)),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(other)?;
                ops_copy(device, copies.iter_mut().map(|(from, to)| (*from, to)))?;
                sides.push(alias::<bf16, Chip, m![Block, Spoken]>(&buffer, ..)?);
            }
            let sum = Function::new(&furiosa_opt_models::ops::pointwise::add::Add, forward)
                .forward(device, (), &mut (), (&sides[0], &sides[1]))
                .await?
                .into_buffer();
            let start = at * Block::SIZE;
            let source = sum.alias(..block.len() * row).map_err(other)?;
            let mut target =
                (output.alias(start * row..(start + block.len()) * row)).map_err(other)?;
            ops_copy(device, [(&source, &mut target)])?;
        }
        Rows::new(&output, 0..pairs.len())
    }

    /// The talker's text rows of `ids`: `text_embedding`, then `text_projection`, a block at a time.
    pub async fn text(
        &self,
        device: &mut Device,
        ids: &[i32],
    ) -> io::Result<Rows<bf16, m![Spoken]>> {
        let embeddings = &self.embeddings;
        let row = Rows::<bf16, m![Spoken]>::bytes();
        let entry = Rows::<bf16, m![TtsTextWidth]>::bytes();
        let output = HbmBuffer::alloc(device, ids.len().max(1) * row).map_err(io::Error::other)?;
        for (at, block) in ids.chunks(Block::SIZE).enumerate() {
            let gathered = gather_rows::<TtsTextVocabulary, TtsTextWidth>(
                device,
                &embeddings.text,
                block,
                entry,
            )
            .await?;
            let inner: HbmTensor<bf16, Chip, m![Block, TtsTextInner]> =
                Function::new(&embeddings.fc1, forward)
                    .forward(device, (), &mut (), &gathered)
                    .await?;
            let inner = Function::new(
                &furiosa_opt_models::ops::pointwise::activation::Silu,
                forward,
            )
            .forward(device, (), &mut (), &inner)
            .await?;
            let projected: HbmTensor<bf16, Chip, m![Block, Spoken]> =
                Function::new(&embeddings.fc2, forward)
                    .forward(device, (), &mut (), &inner)
                    .await?;
            let projected = projected.into_buffer();
            let source = projected
                .alias(..block.len() * row)
                .map_err(io::Error::other)?;
            let start = at * Block::SIZE;
            let mut target = (output.alias(start * row..(start + block.len()) * row))
                .map_err(io::Error::other)?;
            ops_copy(device, [(&source, &mut target)])?;
        }
        Rows::new(&output, 0..ids.len())
    }

    /// The talker's codec table rows of `ids`, which a prompt's control rows read.
    pub async fn codec(
        &self,
        device: &mut Device,
        ids: &[i32],
    ) -> io::Result<Rows<bf16, m![Spoken]>> {
        let row = Rows::<bf16, m![Spoken]>::bytes();
        let output = HbmBuffer::alloc(device, ids.len().max(1) * row).map_err(io::Error::other)?;
        for (at, block) in ids.chunks(Block::SIZE).enumerate() {
            if block.iter().any(|&id| !(0..3072).contains(&id)) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "a codec row outside the table",
                ));
            }
            let gathered =
                gather_rows::<TtsCodecRows, Spoken>(device, &self.embeddings.codecs, block, row)
                    .await?
                    .into_buffer();
            let source = gathered
                .alias(..block.len() * row)
                .map_err(io::Error::other)?;
            let start = at * Block::SIZE;
            let mut target = (output.alias(start * row..(start + block.len()) * row))
                .map_err(io::Error::other)?;
            ops_copy(device, [(&source, &mut target)])?;
        }
        Rows::new(&output, 0..ids.len())
    }
}

/// How a host picks an entry from its FP32 scores: the first highest at temperature 0, else a
/// draw from the softmax of the scores over `temperature`, of the `top_k` highest (all at 0) and
/// of those the fewest whose mass reaches `top_p`, as Transformers samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Sampling {
    /// The first highest score's.
    pub const GREEDY: Self = Self {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
    };

    /// The entry of `scores` this sampling picks with the uniform draw `uniform` in [0, 1).
    pub fn pick(&self, scores: &[f32], uniform: f64) -> usize {
        let first = (0..scores.len()).fold(
            0,
            |best, at| if scores[at] > scores[best] { at } else { best },
        );
        if self.temperature <= 0.0 || scores.len() < 2 {
            return first;
        }
        let temperature = f64::from(self.temperature);
        // Highest first, ties by index: a total order, so the `top_k` it selects and then sorts
        // are the first `top_k` of the whole sorted order.
        let descending = |a: &usize, b: &usize| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b));
        let mut order: Vec<usize> = (0..scores.len()).collect();
        if self.top_k > 0 && self.top_k < order.len() {
            order.select_nth_unstable_by(self.top_k - 1, descending);
            order.truncate(self.top_k);
        }
        order.sort_by(descending);
        let highest = f64::from(scores[order[0]]) / temperature;
        let mut weights: Vec<f64> = (order.iter())
            .map(|&at| (f64::from(scores[at]) / temperature - highest).exp())
            .collect();
        if self.top_p < 1.0 {
            let total: f64 = weights.iter().sum();
            let (mut mass, mut kept) = (0.0, 0);
            while kept < weights.len() && (kept == 0 || mass < f64::from(self.top_p)) {
                mass += weights[kept] / total;
                kept += 1;
            }
            weights.truncate(kept);
        }
        let total: f64 = weights.iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return first;
        }
        let mut target = uniform * total;
        for (at, weight) in weights.iter().enumerate() {
            if target < *weight {
                return order[at];
            }
            target -= weight;
        }
        order[weights.len() - 1]
    }
}

/// A row's uniform draws, SplitMix64 from its seed.
struct Draws(u64);

impl Draws {
    /// The next draw in [0, 1).
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^= mixed >> 31;
        (mixed >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The rows `ids` of `table`, `bytes` a row, in a block of rows; the block's rows past them read
/// the table's first.
async fn gather_rows<R: AxisName + Send + Sync, W: AxisName + Send + Sync>(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![R, W]>,
    ids: &[i32],
    bytes: usize,
) -> io::Result<HbmTensor<bf16, Chip, m![Block, W]>> {
    let mut offsets = vec![0i32; Block::SIZE];
    for (at, &id) in ids.iter().enumerate() {
        let id = usize::try_from(id)
            .ok()
            .filter(|&id| id < R::SIZE)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "an ID outside the table")
            })?;
        offsets[at] = i32::try_from(id * bytes).map_err(io::Error::other)?;
    }
    let bytes: Vec<u8> = offsets
        .iter()
        .flat_map(|offset| offset.to_le_bytes())
        .collect();
    let offsets: HbmTensor<i32, Chip, m![Block]> = alias(
        &HostBufferSlice::from(&bytes[..])
            .to_hbm(&mut device.pdma)
            .await
            .map_err(io::Error::other)?,
        ..,
    )?;
    Function::new(&Gather, forward)
        .forward(device, (), &mut (), (table, &offsets))
        .await
}

/// The BF16 base model of the size `S` on one chip, whose calls a classifier of `S` makes.
type Base<S> = qwen3::Model<S, form::Affine, Parallel<1>, cache::Bf16, Hbm>;

/// The sequence classifiers of the listed Qwen3 sizes `$size`.
macro_rules! classifiers {
    ($($size:ty),+ $(,)?) => {$(
        /// The kept rows' scores, label columns first.
        impl Decoder
            for qwen3::ForSequenceClassification<
                $size,
                form::Affine,
                Parallel<1>,
                cache::Bf16,
                Hbm,
            >
        {
            const OUTPUT: Output = Output::Scores;
            type Output = Rows<f32, m![Columns]>;
            type Layers = <$size as Geometry>::Layers<Blocks<KvHeads, Head>>;
            /// The base model's, of which a classifier's calls run none.
            type Adapter = <Base<$size> as Decoder>::Adapter;

            fn limits(&self) -> Limits {
                Limits {
                    adapters: Self::ADAPTERS,
                    ..self.model.limits()
                }
            }

            fn layers(storage: &[HbmBuffer]) -> io::Result<Self::Layers> {
                views(storage)
            }

            async fn call(
                &self,
                device: &mut Device,
                layers: &mut Self::Layers,
                batch: Batch<'_, Self::Adapter>,
                tiling: Tiling,
            ) -> io::Result<Rows<f32, m![Columns]>> {
                type Size = $size;
                type W = <Size as Geometry>::Hidden;
                type R = <Size as Decode>::Rows;
                let kept = self.model.call(device, layers, batch, tiling).await?;
                if kept.len() == 0 {
                    return Rows::new(kept.buffer(), 0..0);
                }
                let bytes = Rows::<bf16, Symbol<W>>::bytes();
                let views: gather::RowStorage<W, bf16> = (0..kept.len())
                    .map(|row| alias(kept.buffer(), row * bytes..(row + 1) * bytes))
                    .collect::<io::Result<_>>()?;
                let scores = HbmBuffer::alloc(device, R::SIZE * Rows::<f32, m![Columns]>::bytes())
                    .map_err(io::Error::other)?;
                let mut context = affine::Context::<bf16, W, R> {
                    input: gather::Rows::new(device, &views).await?,
                    output: alias(&scores, ..)?,
                };
                let head = Function::new(&self.score, forward);
                head.forward(device, (), &mut (), &mut context).await?;
                Rows::new(&scores, 0..kept.len())
            }
        }
    )+};
}

classifiers!(
    qwen3::Size<"0.6B">,
    qwen3::Size<"1.7B">,
    qwen3::Size<"4B">,
    qwen3::Size<"8B">,
);

/// The kept rows' greedy tokens of each listed launch model `$model` of size `$size`, from one
/// launch of its causal tree: every form of the catalog shares it.
macro_rules! launches {
    ($([$($generics:tt)*] $model:ty, $size:ty);+ $(;)?) => {$(
        impl<$($generics)*> Decoder for $model
        where
            Self: Send + Sync + DeviceSend,
            for<'i, 'a, 's, 'b> Self: Module<
                    &'i mut launch::Input<'a, $size>,
                    (),
                    launch::State<'s, 'b, $size>,
                    Output = (),
                >,
        {
            const OUTPUT: Output = Output::Tokens;
            type Output = Rows<i32, m![1]>;
            type Layers = <$size as Geometry>::Layers<Blocks<KvHeads, Head>>;
            type Adapter = Infallible;

            /// The rows of a launch, a request reaching the positions its rotary table covers at
            /// most.
            fn limits(&self) -> Limits {
                Limits {
                    window: launch::CONTEXT.min(<$size as Decode>::Positions::SIZE),
                    rows: launch::Capacity::<$size>::SIZE,
                    cache: CacheSpec {
                        pools: vec![Pool {
                            block: CacheBlock::SIZE,
                            bytes: Blocks::<KvHeads, Head>::BLOCK_STRIDE as usize,
                        }],
                        groups: vec![Group {
                            pool: 0,
                            layers: 0..<$size as Geometry>::LAYERS,
                            kind: Kind::Paged,
                            window: None,
                        }],
                    },
                    adapters: Self::ADAPTERS,
                    overrides: 0,
                }
            }

            fn layers(storage: &[HbmBuffer]) -> io::Result<Self::Layers> {
                views(storage)
            }

            // The model is captured and launched in place, never held across an await: that `Send`
            // check erases lifetimes and cannot prove the tree's higher-ranked stream bounds.
            fn call(
                &self,
                device: &mut Device,
                layers: &mut Self::Layers,
                batch: Batch<'_, Infallible>,
                tiling: Tiling,
            ) -> impl Future<Output = io::Result<Rows<i32, m![1]>>> + Send {
                let model = self;
                async move {
                    if tiling == Tiling::Fixed {
                        return Err(io::Error::new(
                            io::ErrorKind::Unsupported,
                            "the launch model attends prompt chunks apart from their history",
                        ));
                    }
                    type Capacity = launch::Capacity<$size>;
                    let reads = batch.operands::<
                        Capacity,
                        <$size as Decode>::Reach,
                        <$size as Decode>::Sequences,
                        KvHeads,
                        <$size as Geometry>::Groups,
                        Head,
                        WAVES,
                    >(0)?;
                    let (staged, views) = batch
                        .stage::<Capacity, Tables<Capacity, Chunks>, Chip>(
                            device,
                            &reads,
                            (0, Table::Blocks),
                        )
                        .await?;
                    // ponytail: placed each call; a load-lifetime place would spare two small
                    // uploads a call.
                    let masks = Masks::new(device).await?;
                    let mut views = views.into_iter();
                    let mut next = || {
                        views
                            .next()
                            .ok_or_else(|| io::Error::other("a read array is missing"))
                    };
                    let operands: launch::Operands<$size> = key_value::Operands {
                        batch: key_value::Batch {
                            positions: alias(&next()?, ..)?,
                            slot: alias(&next()?, ..)?,
                        },
                        work: launch::Work::<$size>::bind(&mut next, &masks)?,
                        selected: Selected {
                            live: alias(&next()?, ..)?,
                            rows: alias(&next()?, ..)?,
                        },
                        slots: PhantomData,
                    };
                    let tokens: launch::Ids<$size> = alias(&staged.tokens, ..)?;
                    let mut input: launch::Input<'_, $size> =
                        Input::new(tokens, staged.count, &operands);
                    let slots = layers
                        .each_mut()
                        .map(|blocks| ((Default::default(), blocks, ()), ()));
                    let hidden = DeviceOutput::alloc(device).map_err(io::Error::other)?;
                    let tokens: launch::Tokens<$size> =
                        DeviceOutput::alloc(device).map_err(io::Error::other)?;
                    let mut places = (((hidden, slots), ()), tokens);
                    let state: launch::State<'_, '_, $size> = &mut places;
                    launch(forward, (&mut *device, model, (), state, &mut input))
                        .output(&mut ())
                        .submit()
                        .accepted()
                        .map(drop)
                        .map_err(io::Error::other)?;
                    Rows::new(&places.1.into_buffer(), 0..batch.kept().len())
                }
            }
        }
    )+};
}

// Block-scaled FP8 Qwen3-8B runs its own root, the catalog tree's leaves in the decode order.
launches!(
    [B] launch::Over<qwen3::Size<"0.6B">, B>, qwen3::Size<"0.6B">;
    [B] launch::Over<qwen3::Size<"1.7B">, B>, qwen3::Size<"1.7B">;
    [B] launch::Over<qwen3::Size<"4B">, B>, qwen3::Size<"4B">;
    [B] launch::Over<qwen3::Size<"8B">, B>, qwen3::Size<"8B">;
    [] fp8::ForCausalLM, qwen3::Size<"8B">;
    [] llama::nvfp4::ForCausalLM, llama::nvfp4::Size;
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::{Cache, Instance, Request};
    use furiosa_opt_models::models::decoder::form::Affine;

    /// An instance of the catalog's launch model serves requests on a cache in dense BF16 and in
    /// block-scaled FP8, returning the kept rows' tokens.
    #[test]
    fn launches_serve_requests() {
        fn serves<D>()
        where
            Instance<D>:
                for<'r, 'c> AsyncModule<&'r [Request], (), &'c Cache, Output = Rows<i32, m![1]>>,
        {
        }
        serves::<fp8::ForCausalLM>();
        serves::<launch::Model<qwen3::Size<"8B">, Affine>>();
    }
}
