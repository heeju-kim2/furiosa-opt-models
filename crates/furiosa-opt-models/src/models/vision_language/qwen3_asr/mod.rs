//! Qwen3-ASR (`qwen3_asr`): `Qwen3ASRForConditionalGeneration`'s text model, a Qwen3 causal
//! decoder of a published Qwen3 [`Size`](qwen3::Size), whose audio rows a caller places at the
//! prompt's audio placeholder tokens.
//!
//! - Checkpoint scopes follow Qwen/Qwen3-ASR-1.7B-hf's model.safetensors: the text model at
//!   `model.language_model`, the audio tower at `model.audio_tower` and its projector at
//!   `model.multi_modal_projector`; the head is the text model's token table, which the checkpoint
//!   does not store.
//! - The text model reads its config under `text_config`, a Qwen3 config that its size fixes; it
//!   rotates by the token's index alone, so it serves the text size's calls as they are.
//! - [`ForConditionalGeneration`] reads the text model and its head alone, ignoring the audio tower
//!   and projector, whose rows reach it as overrides of the placeholder rows; a call's override
//!   rows fill the table's last rows, past the 151936 tokens, one a call row.

use std::marker::PhantomData;

use furiosa_opt_std::prelude::*;
use serde::Deserialize;

use crate::load::{Checkpoint, Load, LoadError};
use crate::models::config::{self, Literal};
use crate::models::decoder::Form;
use crate::models::decoder::attention::{Geometry, qwen3};
use crate::models::{Class, Family, Implements};
use crate::specialize::attention::Decode;

pub mod audio;

/// HF `Qwen3ASRForConditionalGeneration` without its audio tower, of the Qwen3 size `S`: the text
/// model, then its greedy head over the token table.
pub type ForConditionalGeneration<S, P, Q, C, X> =
    Conditional<S, qwen3::Model<S, P, Q, C, X>, qwen3::LmHead<S, X>, P>;

/// A conditional checkpoint's text model and the head that reads its rows, of size `G` whose
/// projections hold form `P`, read under `model` with their config under `text_config`.
#[derive(DeviceSend, Load)]
#[load(
    config = "text_config",
    ignore = "audio_tower",
    ignore = "multi_modal_projector"
)]
pub struct Conditional<G: Geometry, Model, Head, P> {
    pub size: PhantomData<G>,
    /// The form of the text model's projections, by which the scope admits a checkpoint.
    pub form: PhantomData<P>,
    pub language_model: Model,
    #[load(tied = "language_model.embed_tokens", config = "tie_word_embeddings")]
    pub lm_head: Head,
}

/// What a Qwen3-ASR config declares of the text model: its type and its Qwen3 config.
#[derive(Deserialize)]
struct Config {
    model_type: Literal<"qwen3_asr">,
    text_config: serde_json::Value,
}

/// The text model's tokens.
#[derive(Deserialize)]
struct Text {
    vocab_size: usize,
}

/// The text model reads a checkpoint under `model` when its config is Qwen3-ASR's, its text config
/// is of the size `G` with tokens that end before the table's override rows, and its form admits
/// the checkpoint.
impl<G: Geometry + Decode + Family, Model, Head, P: Form> Implements
    for Conditional<G, Model, Head, P>
{
    const SIZE: &'static str = G::SIZE;
    const CLASS: Class = Class {
        family: "qwen3_asr",
        name: "ForConditionalGeneration",
    };

    fn scope(source: &Checkpoint) -> Result<&'static str, LoadError> {
        let refusal = |error: std::io::Error| {
            let (name, reason) = ("config.json".into(), error.to_string());
            match error.kind() {
                std::io::ErrorKind::Unsupported => LoadError::Unsupported { name, reason },
                _ => LoadError::InvalidData { name, reason },
            }
        };
        let Config {
            model_type: Literal,
            text_config,
        } = config::parse(source.config()).map_err(refusal)?;
        G::parse(&text_config).map_err(|error| {
            let reason = format!("text_config.{error}");
            refusal(std::io::Error::new(error.kind(), reason))
        })?;
        // A call's override rows fill the table's last rows, one a call row.
        let Text { vocab_size } = config::parse(&text_config).map_err(refusal)?;
        let tokens = G::Vocabulary::SIZE - G::Rows::SIZE;
        if vocab_size > tokens {
            return Err(LoadError::InvalidData {
                name: "config.json".into(),
                reason: format!(
                    "text_config.vocab_size: {vocab_size}, the size holds {tokens} with its \
                     override rows"
                ),
            });
        }
        P::REQUIREMENT.admits(source).map(|()| "model")
    }
}

#[cfg(test)]
mod tests {
    use super::ForConditionalGeneration;
    use crate::load::{Checkpoint, LoadError};
    use crate::models::Implements;
    use crate::models::decoder::attention::qwen3;
    use crate::models::decoder::form;
    use crate::tensor::cache;
    use crate::tensor::memory::Hbm;
    use crate::tensor::parallel::Parallel;
    use serde_json::{Value, json};

    type Text =
        ForConditionalGeneration<qwen3::Size<"1.7B">, form::Affine, Parallel<1>, cache::Bf16, Hbm>;

    /// Qwen/Qwen3-ASR-1.7B-hf's config.json at revision bcd2b5b7f32b480ab5790554cfa8347f246a14f3,
    /// its text model's layer types elided.
    pub(crate) fn published() -> Value {
        json!({
            "architectures": ["Qwen3ASRForConditionalGeneration"],
            "audio_config": {
                "activation_function": "gelu", "d_model": 1024, "downsample_hidden_size": 480,
                "encoder_attention_heads": 16, "encoder_ffn_dim": 4096, "encoder_layers": 24,
                "max_position_embeddings": 13, "model_type": "qwen3_asr_encoder", "n_window": 50,
                "n_window_infer": 800, "num_key_value_heads": 16, "num_mel_bins": 128,
                "output_dim": 2048, "scale_embedding": false
            },
            "audio_token_id": 151676, "dtype": "bfloat16", "eos_token_id": [151643, 151645],
            "model_type": "qwen3_asr", "pad_token_id": 151645,
            "text_config": {
                "attention_bias": false, "head_dim": 128, "hidden_act": "silu",
                "hidden_size": 2048, "intermediate_size": 6144, "max_position_embeddings": 65536,
                "max_window_layers": 28, "model_type": "qwen3", "num_attention_heads": 16,
                "num_hidden_layers": 28, "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
                "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
                "sliding_window": null, "tie_word_embeddings": true, "use_sliding_window": false,
                "vocab_size": 151936
            },
            "tie_word_embeddings": true, "timestamp_token_id": 151705,
        })
    }

    fn scope(config: Value) -> Result<&'static str, LoadError> {
        let empty = tempfile::tempdir().unwrap();
        <Text as Implements>::scope(&Checkpoint::new(empty.path(), Some(config)).unwrap())
    }

    /// The text model reads the published checkpoint under `model`; another model type, a text
    /// config of another size and a feature Qwen3 does not serve are refused by their field.
    #[test]
    fn reads_the_published_config_under_model() {
        assert_eq!(scope(published()).unwrap(), "model");
        let mut other = published();
        other["model_type"] = json!("qwen3_omni_moe");
        let refused = scope(other).unwrap_err().to_string();
        assert!(refused.contains("model_type"), "{refused}");
        let mut wider = published();
        wider["text_config"]["hidden_size"] = json!(1024);
        let refused = scope(wider).unwrap_err();
        assert!(
            matches!(&refused, LoadError::InvalidData { reason, .. }
                if reason == "text_config.hidden_size: 1024, the size 1.7B holds 2048"),
            "{refused:?}"
        );
        let mut biased = published();
        biased["text_config"]["attention_bias"] = json!(true);
        let refused = scope(biased).unwrap_err();
        assert!(
            matches!(&refused, LoadError::Unsupported { reason, .. }
                if reason == "text_config.attention_bias: Qwen3 projects without biases"),
            "{refused:?}"
        );
    }
}
