//! Qwen3-TTS (`qwen3_tts`), by published size: the talker of `Qwen3TTSForConditionalGeneration`,
//! a Qwen3 decoder over codec frames whose greedy head picks each frame's first codebook entry.
//!
//! - Checkpoint scopes follow Qwen/Qwen3-TTS-12Hz-1.7B-Base's model.safetensors: the talker at
//!   `talker`, its body at `talker.model` with the codec table `codec_embedding`, its head at
//!   `talker.codec_head`; the talker's config is `talker_config`.
//! - A talker row's input is a sum of embeddings the caller computes: the projected text row and
//!   every codebook's entry of the frame before. Every row reaches the body as an override, so the
//!   body's own table is read only by a row a caller leaves unreplaced.
//! - Every row rotates by its index alone: a talker prompt gives the three interleaved sections
//!   one position, so the talker serves Qwen3's calls of its geometry as they are.
//! - The head scores the 2048 codebook entries and the end of speech alone: Transformers' talker
//!   generation suppresses the other control rows, the last 1024, at every step. The table
//!   compiles 256 rows past the talker's 3072, which no token reads and a call's override rows
//!   fill.
//! - [`ForConditionalGeneration`] reads the talker's body and head alone; the text table and
//!   projection, the code predictor, the speaker encoder and the speech tokenizer are not read.

use std::io;
use std::marker::PhantomData;

use furiosa_opt_std::prelude::*;
use serde::Deserialize;

use super::qwen3::{self, Front};
use super::{Geometry, Head, KvHeads};
use crate::load::{Checkpoint, Children, Context, Load, LoadError, Named};
use crate::models::config::Literal;
use crate::models::decoder::Form;
use crate::models::{Class, Family, Implements};
use crate::modules::attention::NormalizedQkv;
use crate::modules::language::Greedy;
use crate::ops::attention::geometry::{self, ProjectionWidth, QueryGroups, QueryWidth};
use crate::ops::linear::Affine;
use crate::ops::normalization::RmsNorm;
use crate::ops::swiglu;
use crate::specialize::attention::Decode;
use crate::tensor::axis;
use crate::tensor::form::Weight;
use crate::tensor::memory::Memory;
use crate::tensor::parallel::Chip;

pub mod codec;
pub mod speaker;

/// HF `Qwen3TTSTalkerForConditionalGeneration` of the talker size `S`, read under `talker`: the
/// body, then the greedy codec head.
pub type ForConditionalGeneration<S, P, Q, C, X> =
    Conditional<S, qwen3::Model<S, P, Q, C, X>, qwen3::LmHead<S, X>, P>;

/// A talker's body and the head that reads its rows, of size `G` whose projections hold form `P`,
/// their config under `talker_config`.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct Conditional<G: Geometry, Model, Head, P> {
    pub size: PhantomData<G>,
    /// The form of the body's projections, by which the scope admits a checkpoint.
    pub form: PhantomData<P>,
    pub model: Model,
    pub codec_head: Head,
}

/// The body under `model`, its text table ignored, and the head under `codec_head` with its control
/// rows suppressed; the talker's other modules are ignored.
impl<'a, G: Geometry, Model, Head, P> Load<&'a Checkpoint> for Conditional<G, Model, Head, P>
where
    Model: Load<&'a Checkpoint>,
    Head: Load<&'a Checkpoint> + Suppress,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            for unread in ["model.text_embedding", "text_projection", "code_predictor"] {
                cx.ignore(unread);
            }
            cx.config("talker_config", |cx| {
                let mut children = Children::default();
                let model = cx.field("model", "model", |cx| Model::load(cx, from));
                let model = children.ask(model)?;
                let head = cx.field("codec_head", "codec_head", |cx| {
                    let head = Head::load(cx, from)?;
                    let (_, eos) = cx.config_number(from, "codec_eos_token_id")?;
                    head.suppress(cx, |row| row < CODEBOOK || row == eos as usize)
                });
                let head = children.ask(head)?;
                children
                    .node(model.zip(head))
                    .map(|(model, codec_head)| Self {
                        size: PhantomData,
                        form: PhantomData,
                        model,
                        codec_head,
                    })
            })
        })
    }
}

/// A head whose scores of some rows can be lowered below every other's.
pub trait Suppress: Sized {
    /// The head with the rows `kept` refuses lowered to the lowest finite score.
    fn suppress(
        self,
        context: &mut Context<'_>,
        kept: impl Fn(usize) -> bool,
    ) -> Result<Self, LoadError>;
}

/// The bias of a greedy head, rebuilt: zero for a kept row, the lowest finite score for another.
impl<H: AxisName, V: AxisName, W, B, S> Suppress for Greedy<H, V, Affine<W, B>, S>
where
    B: Load<HostTensor<f32, m![{ Chip }, { m![V] }]>>,
{
    fn suppress(
        mut self,
        context: &mut Context<'_>,
        kept: impl Fn(usize) -> bool,
    ) -> Result<Self, LoadError> {
        let bias = (0..V::SIZE).map(|row| if kept(row) { 0.0 } else { f32::MIN });
        let bias = HostTensor::from_vec(bias.collect::<Vec<_>>());
        self.head.bias = context.field("head", "", |cx| {
            cx.field("bias", "bias", |cx| B::load(cx, bias))
        })?;
        Ok(self)
    }
}

/// A published Qwen3-TTS talker size, by its hub spelling.
pub struct Talker<const N: &'static str>;

/// The talker of every size: Qwen3's front, its token table the codec table.
impl<const N: &'static str> Front for Talker<N> {
    type Heads<Projection, Norm> =
        NormalizedQkv<Projection, Named<"q_norm", Norm>, Named<"k_norm", Norm>>;
    type HeadNorm<Norm> = Norm;
    type Table<Table> = Named<"codec_embedding", Table>;
    type Tokens<Vocabulary: AxisName> = Vocabulary;
}

// The talker's table as compiled: its 3072 rows, then the rows a call's override rows fill.
axes![Codec = 3328];

/// The entries of a codebook, the rows before the talker's control rows.
const CODEBOOK: usize = 2048;

/// The talker's rows: 2048 entries of each codebook and 1024 control rows.
const ROWS: usize = 3072;

/// The talker's norm: `rms_norm_eps` 1e-6.
type TalkerNorm<W, X> =
    RmsNorm<W, { EPSILON }, m![1], Weight<m![W], m![1], <X as Memory>::Tensor<bf16, m![1], m![W]>>>;

const EPSILON: usize = 1e-6_f32.to_bits() as usize;

impl Geometry for Talker<"1.7B"> {
    const SIZE: &'static str = "1.7B";
    const FAMILY: &'static str = "qwen3_tts";
    type Queries = QueryWidth<2048>;
    type Hidden = geometry::Hidden<2048>;
    type Intermediate = swiglu::Intermediate<6144>;
    type Vocabulary = Codec;
    type KvHeads = KvHeads;
    type Groups = QueryGroups<2>;
    type Head = Head;
    type Layers<L> = [L; Self::LAYERS];
    type Norm<W: AxisName, X: Memory> = TalkerNorm<W, X>;

    const LAYERS: usize = 28;
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = false;
}

/// Qwen3-1.7B's calls: the talker's layers are of its geometry.
impl Decode for Talker<"1.7B"> {
    type Rows = <qwen3::Size<"1.7B"> as Decode>::Rows;
    type Windows = <qwen3::Size<"1.7B"> as Decode>::Windows;
    type Width = ProjectionWidth<4096>;
    type Packed = m![{ Symbol<ProjectionWidth<4096>> }, { Symbol<geometry::Hidden<2048>> }];
    type Work = <qwen3::Size<"1.7B"> as Decode>::Work;
    type Capacity = axis::Rows<1024>;
    type Reach = axis::Rows<2048>;
    type Sequences = axis::Rows<8>;
    type Slots = <qwen3::Size<"1.7B"> as Decode>::Slots;
    type Positions = axis::Rows<32768>;
}

/// What a Qwen3-TTS config declares that the talker's size fixes, and the features it does not
/// serve.
#[derive(Deserialize)]
pub struct Config {
    model_type: Literal<"qwen3_tts">,
    talker_config: TalkerConfig,
}

/// The talker's part of a Qwen3-TTS config.
#[derive(Deserialize)]
struct TalkerConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    codec_eos_token_id: usize,
    #[serde(default)]
    rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    use_sliding_window: bool,
}

/// The rotation's sections, interleaved across the head.
#[derive(Deserialize)]
struct RopeScaling {
    #[serde(default)]
    interleaved: bool,
    #[serde(default)]
    rope_type: Option<String>,
}

/// A config is of a size when its talker declares the size's values, its end of speech among the
/// control rows and none of the features the talker does not serve.
impl<const N: &'static str> TryFrom<Config> for Talker<N>
where
    Self: Geometry,
{
    type Error = io::Error;

    fn try_from(config: Config) -> io::Result<Self> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        let unsupported = |reason: &str| io::Error::new(io::ErrorKind::Unsupported, reason);
        let Config {
            model_type: Literal,
            talker_config: talker,
        } = config;
        let heads = <Self as Geometry>::KvHeads::SIZE;
        let fixed = [
            (
                "hidden_size",
                talker.hidden_size,
                <Self as Geometry>::Hidden::SIZE,
            ),
            (
                "intermediate_size",
                talker.intermediate_size,
                <Self as Geometry>::Intermediate::SIZE,
            ),
            ("num_hidden_layers", talker.num_hidden_layers, Self::LAYERS),
            ("num_key_value_heads", talker.num_key_value_heads, heads),
            (
                "num_attention_heads",
                talker.num_attention_heads,
                heads * <Self as Geometry>::Groups::SIZE,
            ),
            ("head_dim", talker.head_dim, <Self as Geometry>::Head::SIZE),
            ("vocab_size", talker.vocab_size, ROWS),
        ];
        if let Some((key, declared, held)) = fixed
            .into_iter()
            .find(|(_, declared, held)| declared != held)
        {
            return Err(invalid(format!(
                "talker_config.{key}: {declared}, the size {N} holds {held}"
            )));
        }
        if (talker.rms_norm_eps as f32).to_bits() as usize != Self::EPSILON {
            return Err(invalid(format!(
                "talker_config.rms_norm_eps: {}, another than the size's",
                talker.rms_norm_eps
            )));
        }
        if !(CODEBOOK..ROWS).contains(&talker.codec_eos_token_id) {
            return Err(invalid(format!(
                "talker_config.codec_eos_token_id: {}, not a control row",
                talker.codec_eos_token_id
            )));
        }
        let interleaved = |rotation: &RopeScaling| {
            rotation.interleaved && rotation.rope_type.as_deref().is_none_or(|t| t == "default")
        };
        match (
            talker.rope_scaling,
            talker.attention_bias,
            talker.use_sliding_window,
        ) {
            (Some(rotation), ..) if !interleaved(&rotation) => Err(unsupported(
                "talker_config.rope_scaling: the talker rotates its interleaved sections alone",
            )),
            (_, true, _) => Err(unsupported(
                "talker_config.attention_bias: the talker projects without biases",
            )),
            (.., true) => Err(unsupported(
                "talker_config.use_sliding_window: the talker attends the whole history",
            )),
            _ => Ok(Talker),
        }
    }
}

impl<const N: &'static str> Family for Talker<N>
where
    Self: Geometry,
{
    type Hf = Config;
}

/// The talker reads a checkpoint under `talker` when its config is of the size `G` and its form
/// admits the checkpoint.
impl<G: Geometry + Family, Model, Head, P: Form> Implements for Conditional<G, Model, Head, P> {
    const SIZE: &'static str = G::SIZE;
    const CLASS: Class = Class {
        family: "qwen3_tts",
        name: "ForConditionalGeneration",
    };

    fn scope(source: &Checkpoint) -> Result<&'static str, LoadError> {
        G::parse(source.config()).map_err(|error| {
            let (name, reason) = ("config.json".into(), error.to_string());
            match error.kind() {
                io::ErrorKind::Unsupported => LoadError::Unsupported { name, reason },
                _ => LoadError::InvalidData { name, reason },
            }
        })?;
        P::REQUIREMENT.admits(source).map(|()| "talker")
    }
}

/// HF `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration` of the predictor size `S`, read
/// under `talker` with the talker's codec table: a frame's input rows from their tables, projected
/// into the body's width, the body, then each step's greedy head.
pub type CodePredictor<S, P, Q, C, X> = Codes<
    S,
    qwen3::Model<S, P, Q, C, X>,
    qwen3::LmHead<S, X>,
    Biased<Text, geometry::Hidden<1024>, X>,
    <X as Memory>::Tensor<bf16, Chip, m![Talked, Text]>,
    <X as Memory>::Tensor<bf16, Chip, m![Codebook, Text]>,
    P,
>;

/// The code predictor of a frame's other codebooks: the talker's table of each frame's first
/// entry, the projection of its 2048-feature input rows into the body's width, the body, whose
/// token table holds the projected rows, and each later codebook's table and head.
///
/// - Row 0 of a frame's sequence is the talker's hidden row, which the caller gives; row 1 is the
///   talker's table row of the frame's first entry; row `q > 1` is table `q - 2`'s row of entry
///   `q - 1`. Row `q > 0` picks entry `q` by head `q - 1`.
/// - The body's table holds no checkpoint tensor: its rows are zero but for those a call writes,
///   from row 2048 on, which its rows' token IDs name.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct Codes<G: Geometry, Model, Head, Projection, Talker, Table, P> {
    pub size: PhantomData<G>,
    /// The form of the body's projections, by which the scope admits a checkpoint.
    pub form: PhantomData<P>,
    pub talker_table: Talker,
    pub small_to_mtp_projection: Projection,
    pub model: Model,
    pub tables: [Table; GROUPS],
    pub lm_head: [Head; GROUPS],
}

/// The talker's codec table under `model.codec_embedding`; the projection, the tables and heads
/// under `code_predictor`, with their config under `code_predictor_config`, and the body there
/// with a zero token table of the checkpoint's `vocab_size` rows; the talker's other modules are
/// ignored.
impl<'a, G: Geometry, Model, Head, Projection, Talker, Table, P> Load<&'a Checkpoint>
    for Codes<G, Model, Head, Projection, Talker, Table, P>
where
    Model: for<'b> Load<&'b Checkpoint>,
    Head: Load<&'a Checkpoint>,
    Projection: Load<&'a Checkpoint>,
    Talker: Load<&'a Checkpoint>,
    Table: Load<&'a Checkpoint>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            for unread in [
                "model.layers",
                "model.norm",
                "model.text_embedding",
                "codec_head",
                "text_projection",
            ] {
                cx.ignore(unread);
            }
            let mut children = Children::default();
            let talker = cx.field("talker_table", "model.codec_embedding.weight", |cx| {
                Talker::load(cx, from)
            });
            let talker = children.ask(talker)?;
            let predicted = cx.config("talker_config", |cx| {
                cx.config("code_predictor_config", |cx| {
                    cx.field("", "code_predictor", |cx| predictor(cx, from))
                })
            });
            let predicted = children.ask(predicted)?;
            children.node(talker.zip(predicted)).map(
                |(talker_table, (projection, model, tables, lm_head))| Self {
                    size: PhantomData,
                    form: PhantomData,
                    talker_table,
                    small_to_mtp_projection: projection,
                    model,
                    tables,
                    lm_head,
                },
            )
        })
    }
}

/// The talker and its code predictor together, as vLLM-Omni's `Qwen3TTSTalkerForConditionalGeneration`
/// runs them a frame at a time: a talker row's scores of its frame's first entry, the frame's other
/// entries by the predictor fed that row, and the rows a prompt and a frame give the talker.
pub struct Speaker<Talker, Predictor> {
    pub talker: Talker,
    pub predictor: Predictor,
    pub embeddings: Embeddings,
}

axes![
    TextVocabulary = 151936,
    TextWidth = 2048,
    TextInner = 2048,
    CodecRows = 33792,
    Stacked = 32768
];

/// The rows a talker reads: `model.text_embedding` and `text_projection`, a SiLU pair into the
/// talker's width; the talker's codec table and each later codebook's, stacked; and the projection
/// that sums a frame's sixteen rows of them in FP32, rounding once.
#[derive(DeviceSend)]
pub struct Embeddings {
    pub text: HbmTensor<bf16, Chip, m![TextVocabulary, TextWidth]>,
    pub fc1: Affine<
        HbmTensor<bf16, Chip, m![TextInner, TextWidth]>,
        HbmTensor<f32, Chip, m![TextInner]>,
    >,
    pub fc2: Affine<HbmTensor<bf16, Chip, m![Text, TextInner]>, HbmTensor<f32, Chip, m![Text]>>,
    /// The talker's 3072 rows, then each later codebook's 2048.
    pub codecs: HbmTensor<bf16, Chip, m![CodecRows, Text]>,
    /// Ones where an output feature is its input row's feature, zero elsewhere.
    pub sum: crate::ops::linear::Linear<Stacked, Text, Chip>,
}

impl<'a> Load<&'a Checkpoint> for Embeddings {
    fn load(cx: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        let width = Text::SIZE;
        let rows = |cx: &mut Context<'_>, path: &str, count: usize| {
            let read = cx.field("", path, |cx| cx.floats(from, &[count, width]))?;
            Ok::<_, LoadError>(read.into_iter().map(bf16::from_f32).collect::<Vec<_>>())
        };
        let text = cx.field("text", "model.text_embedding.weight", |cx| {
            HbmTensor::load(cx, from)
        })?;
        let fc1 = cx.field("fc1", "text_projection.linear_fc1", |cx| {
            Affine::load(cx, from)
        })?;
        let fc2 = cx.field("fc2", "text_projection.linear_fc2", |cx| {
            Affine::load(cx, from)
        })?;
        let mut codecs = rows(cx, "model.codec_embedding.weight", ROWS)?;
        for group in 0..GROUPS {
            let path = format!("code_predictor.model.codec_embedding.{group}.weight");
            codecs.extend(rows(cx, &path, CODEBOOK)?);
        }
        let codecs = HbmTensor::load(
            cx,
            HostTensor::<bf16, m![CodecRows, Text]>::from_vec(codecs),
        )?;
        let ones: Vec<bf16> = (0..Text::SIZE * Stacked::SIZE)
            .map(|at| bf16::from_f32(f32::from(at / Stacked::SIZE == at % Stacked::SIZE % width)))
            .collect();
        let sum = crate::ops::linear::Linear::new(Weight::new(HbmTensor::load(
            cx,
            HostTensor::<bf16, m![Text, Stacked]>::from_vec(ones),
        )?));
        Ok(Self {
            text,
            fc1,
            fc2,
            codecs,
            sum,
        })
    }
}

/// The talker, the predictor and the rows they read at the current name, in one strict traversal:
/// what one ignores another reads.
impl<'a, T: Load<&'a Checkpoint>, P: Load<&'a Checkpoint>> Load<&'a Checkpoint> for Speaker<T, P> {
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let talker = T::load(cx, from)?;
            let predictor = P::load(cx, from)?;
            let embeddings = Embeddings::load(cx, from)?;
            Ok(Self {
                talker,
                predictor,
                embeddings,
            })
        })
    }
}

/// The speaker reads a checkpoint where its talker does.
impl<T: Implements, P> Implements for Speaker<T, P> {
    const SIZE: &'static str = T::SIZE;
    const CLASS: Class = Class {
        family: "qwen3_tts",
        name: "Speaker",
    };

    fn scope(source: &Checkpoint) -> Result<&'static str, LoadError> {
        T::scope(source)
    }
}

/// The codebooks a predictor picks, one a head: each frame's but its first.
pub const GROUPS: usize = 15;

/// A published Qwen3-TTS code predictor size, by its talker's hub spelling.
pub struct Predictor<const N: &'static str>;

/// The predictor of every size: Qwen3's front, its token table the rows its calls read.
impl<const N: &'static str> Front for Predictor<N> {
    type Heads<Projection, Norm> =
        NormalizedQkv<Projection, Named<"q_norm", Norm>, Named<"k_norm", Norm>>;
    type HeadNorm<Norm> = Norm;
    type Table<Table> = Named<"rows", Table>;
    type Tokens<Vocabulary: AxisName> = PredictorRows;
}

// The predictor's head as compiled: a codebook's entries, then 256 more, which round its passes;
// a codebook's and the talker's stored rows; the talker's width, the projection's input.
axes![Entries = 2304, Codebook = 2048, Talked = 3072, Text = 2048];

/// Rows of the predictor's token table: every entry of the talker's table and of the tables of
/// positions 2 to 15 projected into the body's width, one table after another, then 256 rows,
/// whose first 128 a call's own rows fill.
pub const PROJECTED: usize = Talked::SIZE + (GROUPS - 1) * Codebook::SIZE;
axes![PredictorRows = 32000];
const _: () = assert!(PredictorRows::SIZE == PROJECTED + 256);

/// BF16 coefficients over `In` with an FP32 bias of `Out`, in memory `X`.
type Biased<In, Out, X> = Affine<
    <X as Memory>::Tensor<bf16, Chip, m![Out, In]>,
    <X as Memory>::Tensor<f32, Chip, m![Out]>,
>;

impl Geometry for Predictor<"1.7B"> {
    const SIZE: &'static str = "1.7B";
    const FAMILY: &'static str = "qwen3_tts";
    type Queries = QueryWidth<2048>;
    type Hidden = geometry::Hidden<1024>;
    type Intermediate = swiglu::Intermediate<3072>;
    type Vocabulary = Entries;
    type KvHeads = KvHeads;
    type Groups = QueryGroups<2>;
    type Head = Head;
    type Layers<L> = [L; Self::LAYERS];
    type Norm<W: AxisName, X: Memory> = TalkerNorm<W, X>;

    const LAYERS: usize = 5;
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = false;
}

/// Qwen3-0.6B's calls: the predictor's layers are of its geometry.
impl Decode for Predictor<"1.7B"> {
    type Rows = <qwen3::Size<"0.6B"> as Decode>::Rows;
    type Windows = <qwen3::Size<"0.6B"> as Decode>::Windows;
    type Width = ProjectionWidth<4096>;
    type Packed = m![{ Symbol<ProjectionWidth<4096>> }, { Symbol<geometry::Hidden<1024>> }];
    type Work = <qwen3::Size<"0.6B"> as Decode>::Work;
    type Capacity = axis::Rows<1024>;
    type Reach = axis::Rows<2048>;
    type Sequences = axis::Rows<8>;
    type Slots = <qwen3::Size<"0.6B"> as Decode>::Slots;
    type Positions = axis::Rows<32768>;
}

/// The predictor's part of a Qwen3-TTS config: the talker's width and the predictor's config.
#[derive(Deserialize)]
pub struct PredictorHf {
    model_type: Literal<"qwen3_tts">,
    talker_config: TalkerHf,
}

#[derive(Deserialize)]
struct TalkerHf {
    hidden_size: usize,
    num_code_groups: usize,
    code_predictor_config: PredictorConfig,
}

#[derive(Deserialize)]
struct PredictorConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    num_code_groups: usize,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    use_sliding_window: bool,
}

/// A config is of a predictor size when it declares the size's values and a codebook a head.
impl<const N: &'static str> TryFrom<PredictorHf> for Predictor<N>
where
    Self: Geometry,
{
    type Error = io::Error;

    fn try_from(config: PredictorHf) -> io::Result<Self> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        let unsupported = |reason: &str| io::Error::new(io::ErrorKind::Unsupported, reason);
        let PredictorHf {
            model_type: Literal,
            talker_config: talker,
        } = config;
        let predictor = talker.code_predictor_config;
        let heads = <Self as Geometry>::KvHeads::SIZE;
        let fixed = [
            (
                "hidden_size",
                predictor.hidden_size,
                <Self as Geometry>::Hidden::SIZE,
            ),
            (
                "intermediate_size",
                predictor.intermediate_size,
                <Self as Geometry>::Intermediate::SIZE,
            ),
            (
                "num_hidden_layers",
                predictor.num_hidden_layers,
                Self::LAYERS,
            ),
            ("num_key_value_heads", predictor.num_key_value_heads, heads),
            (
                "num_attention_heads",
                predictor.num_attention_heads,
                heads * <Self as Geometry>::Groups::SIZE,
            ),
            (
                "head_dim",
                predictor.head_dim,
                <Self as Geometry>::Head::SIZE,
            ),
            ("vocab_size", predictor.vocab_size, Codebook::SIZE),
            ("num_code_groups", predictor.num_code_groups, GROUPS + 1),
        ];
        if let Some((key, declared, held)) = fixed
            .into_iter()
            .find(|(_, declared, held)| declared != held)
        {
            return Err(invalid(format!(
                "talker_config.code_predictor_config.{key}: {declared}, the size {N} holds {held}"
            )));
        }
        if (talker.hidden_size, talker.num_code_groups) != (Text::SIZE, GROUPS + 1) {
            return Err(invalid(format!(
                "talker_config: hidden_size {} and num_code_groups {}, the size {N} holds {} and {}",
                talker.hidden_size,
                talker.num_code_groups,
                Text::SIZE,
                GROUPS + 1
            )));
        }
        if (predictor.rms_norm_eps as f32).to_bits() as usize != Self::EPSILON {
            return Err(invalid(format!(
                "talker_config.code_predictor_config.rms_norm_eps: {}, another than the size's",
                predictor.rms_norm_eps
            )));
        }
        match (predictor.attention_bias, predictor.use_sliding_window) {
            (true, _) => Err(unsupported(
                "talker_config.code_predictor_config.attention_bias: the predictor projects \
                 without biases",
            )),
            (_, true) => Err(unsupported(
                "talker_config.code_predictor_config.use_sliding_window: the predictor attends \
                 the whole history",
            )),
            _ => Ok(Predictor),
        }
    }
}

impl<const N: &'static str> Family for Predictor<N>
where
    Self: Geometry,
{
    type Hf = PredictorHf;
}

/// The predictor reads a checkpoint under `talker` when its config is of the size `G` and its form
/// admits the checkpoint.
impl<G: Geometry + Family, Model, Head, Projection, Talker, Table, P: Form> Implements
    for Codes<G, Model, Head, Projection, Talker, Table, P>
{
    const SIZE: &'static str = G::SIZE;
    const CLASS: Class = Class {
        family: "qwen3_tts",
        name: "CodePredictor",
    };

    fn scope(source: &Checkpoint) -> Result<&'static str, LoadError> {
        G::parse(source.config()).map_err(|error| {
            let (name, reason) = ("config.json".into(), error.to_string());
            match error.kind() {
                io::ErrorKind::Unsupported => LoadError::Unsupported { name, reason },
                _ => LoadError::InvalidData { name, reason },
            }
        })?;
        P::REQUIREMENT.admits(source).map(|()| "talker")
    }
}

/// The first row of each codebook's table in [`Embeddings::codecs`]: the talker's first.
pub const fn codec_row(group: usize) -> usize {
    match group {
        0 => 0,
        _ => ROWS + (group - 1) * CODEBOOK,
    }
}

/// The predictor's projection, body, tables and heads at the current name.
fn predictor<'a, Model, Head, Projection, Table>(
    cx: &mut Context<'_>,
    from: &'a Checkpoint,
) -> Result<(Projection, Model, [Table; GROUPS], [Head; GROUPS]), LoadError>
where
    Model: for<'b> Load<&'b Checkpoint>,
    Head: Load<&'a Checkpoint>,
    Projection: Load<&'a Checkpoint>,
    Table: Load<&'a Checkpoint>,
{
    let mut children = Children::default();
    let projection = cx.field("small_to_mtp_projection", "small_to_mtp_projection", |cx| {
        Projection::load(cx, from)
    });
    let projection = children.ask(projection)?;
    let model = cx.field("model", "model", |cx| {
        let (_, rows) = cx.config_number(from, "vocab_size")?;
        let (_, width) = cx.config_number(from, "hidden_size")?;
        let shape = [rows as usize, width as usize];
        let zeros = vec![0; shape[0] * shape[1] * size_of::<bf16>()];
        let name = format!("{}.rows.weight", cx.name());
        let extended = from.with(&name, safetensors::Dtype::BF16, &shape, zeros)?;
        Model::load(cx, &extended)
    });
    let model = children.ask(model)?;
    let tables = cx.field("tables", "model.codec_embedding", |cx| {
        array(|at| {
            let name = at.to_string();
            cx.field(&name, &format!("{at}.weight"), |cx| Table::load(cx, from))
        })
    });
    let tables = children.ask(tables)?;
    let heads = cx.field("lm_head", "lm_head", |cx| {
        array(|at| {
            let name = at.to_string();
            cx.field(&name, &name, |cx| Head::load(cx, from))
        })
    });
    let heads = children.ask(heads)?;
    let parts = projection.zip(model).zip(tables).zip(heads);
    children
        .node(parts)
        .map(|(((projection, model), tables), heads)| (projection, model, tables, heads))
}

/// The `GROUPS` values `read` gives, each at its index; the names they lack are reported together.
fn array<T>(mut read: impl FnMut(usize) -> Result<T, LoadError>) -> Result<[T; GROUPS], LoadError> {
    let mut children = Children::default();
    let mut read = (0..GROUPS)
        .map(|at| children.ask(read(at)))
        .collect::<Result<Vec<_>, _>>()?;
    let values: Option<Vec<T>> = read.drain(..).collect();
    children.node(values).map(|values| {
        values
            .try_into()
            .unwrap_or_else(|_| unreachable!("one value an index"))
    })
}

#[cfg(test)]
mod tests {
    use super::{Config, Talker};
    use crate::models::Family;
    use crate::models::config;
    use serde_json::{Value, json};
    use std::io;

    /// The talker's part of Qwen/Qwen3-TTS-12Hz-1.7B-Base's config.json at revision
    /// fd4b254389122332181a7c3db7f27e918eec64e3, its code predictor and codec ids elided.
    fn published() -> Value {
        json!({
            "architectures": ["Qwen3TTSForConditionalGeneration"], "model_type": "qwen3_tts",
            "talker_config": {
                "attention_bias": false, "codec_eos_token_id": 2150, "head_dim": 128,
                "hidden_act": "silu", "hidden_size": 2048, "intermediate_size": 6144,
                "max_position_embeddings": 32768, "model_type": "qwen3_tts_talker",
                "num_attention_heads": 16, "num_code_groups": 16, "num_hidden_layers": 28,
                "num_key_value_heads": 8, "position_id_per_seconds": 13, "rms_norm_eps": 1e-06,
                "rope_scaling": {
                    "interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default",
                    "type": "default"
                },
                "rope_theta": 1000000, "text_hidden_size": 2048, "text_vocab_size": 151936,
                "use_sliding_window": false, "vocab_size": 3072
            },
        })
    }

    /// The 1.7B talker reads its published config; a value the size fixes, an end of speech
    /// outside the control rows and a feature the talker does not serve are refused by field.
    #[test]
    fn talker_refuses_what_it_does_not_serve() {
        let parse = |document: &Value| Talker::<"1.7B">::parse(document).map(drop);
        parse(&published()).unwrap();
        config::parse::<Config>(&published()).map(drop).unwrap();
        use io::ErrorKind::{InvalidData, Unsupported};
        for (key, value, kind, reason) in [
            (
                "hidden_size",
                json!(1024),
                InvalidData,
                "talker_config.hidden_size: 1024, the size 1.7B holds 2048",
            ),
            (
                "vocab_size",
                json!(4096),
                InvalidData,
                "talker_config.vocab_size: 4096, the size 1.7B holds 3072",
            ),
            (
                "codec_eos_token_id",
                json!(5),
                InvalidData,
                "talker_config.codec_eos_token_id: 5, not a control row",
            ),
            (
                "attention_bias",
                json!(true),
                Unsupported,
                "talker_config.attention_bias: the talker projects without biases",
            ),
            (
                "rope_scaling",
                json!({"interleaved": false, "mrope_section": [24, 20, 20]}),
                Unsupported,
                "talker_config.rope_scaling: the talker rotates its interleaved sections alone",
            ),
        ] {
            let mut document = published();
            document["talker_config"][key] = value.clone();
            let error = parse(&document).unwrap_err();
            assert_eq!(
                (error.kind(), error.to_string()),
                (kind, reason.to_string()),
                "{key}: {value}"
            );
        }
    }
}
