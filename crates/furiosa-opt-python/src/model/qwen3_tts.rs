//! The Qwen3-TTS class a load of a Qwen3-TTS checkpoint returns: `ForConditionalGeneration`, the
//! talker, a `Decoder` whose calls return each kept row's greedy first codebook entry, every input
//! row an override the caller sums.

use super::{Constructor, Decoder};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3_stub_gen::derive::gen_stub_pyclass;

/// Adds the Qwen3-TTS class to `module`.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ForConditionalGeneration>()
}

/// The Qwen3-TTS class by name, as the constructor of its objects.
pub(super) const CLASSES: &[(&str, Constructor)] = &[(
    "ForConditionalGeneration",
    Decoder::object::<ForConditionalGeneration>,
)];

/// HF `Qwen3TTSTalkerForConditionalGeneration`, loaded on a device: the talker's body, then its
/// greedy codec head over the codebook entries and the end of speech.
#[gen_stub_pyclass]
#[pyclass(frozen, extends = Decoder, module = "engine.qwen3_tts")]
#[derive(Default)]
pub(super) struct ForConditionalGeneration;
