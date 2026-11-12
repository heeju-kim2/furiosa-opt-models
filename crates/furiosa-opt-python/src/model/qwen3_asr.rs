//! The Qwen3-ASR class a load of a Qwen3-ASR checkpoint returns: `ForConditionalGeneration`, a
//! `Decoder` whose calls return the greedy token of each kept row, its audio rows placed by the
//! caller.

use super::{Constructor, Decoder};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3_stub_gen::derive::gen_stub_pyclass;

/// Adds the Qwen3-ASR class to `module`.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ForConditionalGeneration>()
}

/// The Qwen3-ASR class by name, as the constructor of its objects.
pub(super) const CLASSES: &[(&str, Constructor)] = &[(
    "ForConditionalGeneration",
    Decoder::object::<ForConditionalGeneration>,
)];

/// HF `Qwen3ASRForConditionalGeneration`'s text model, loaded on a device: the Qwen3 base model,
/// then its greedy vocabulary head; a request's audio rows override its placeholder rows.
#[gen_stub_pyclass]
#[pyclass(frozen, extends = Decoder, module = "engine.qwen3_asr")]
#[derive(Default)]
pub(super) struct ForConditionalGeneration;
