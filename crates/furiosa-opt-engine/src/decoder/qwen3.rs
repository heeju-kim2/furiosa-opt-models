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
use super::{Batch, CacheSpec, Decoder, Group, Instance, Kind, Limits, Output, Pool, Rows, Table};
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
        let first = qwen3_tts::Codebook::SIZE;
        let width = Rows::<bf16, Symbol<W>>::bytes();
        let mut placed = (self
            .table
            .alias(first * width..(first + Block::SIZE) * width))
        .map_err(other)?;
        let projected = projected.into_buffer();
        ops_copy(device, [(&projected, &mut placed)])?;
        let batch = batch.named(device, first).await?;
        let kept = self
            .decoder
            .model
            .call(device, layers, batch, tiling)
            .await?;
        greedy!(Predictor, &self.decoder.lm_head[step - 1], device, kept)
    }
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
