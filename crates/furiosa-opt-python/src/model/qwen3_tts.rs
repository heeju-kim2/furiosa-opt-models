//! The Qwen3-TTS classes: `ForConditionalGeneration`, the talker, a `Decoder` whose calls return
//! each kept row's greedy first codebook entry, every input row an override the caller sums;
//! `Speaker`, the talker and its code predictor, a `Decoder` whose calls return each kept row's
//! scores of its frame's first entry, and whose methods give the frames of the entries the caller
//! picked and the talker's input rows; `SpeakerEncoder`, whose calls turn a reference clip's mel
//! frames into its speaker embedding; and `SpeechEncoder`, the speech tokenizer's encoder, whose
//! calls turn a reference clip into its codec frames. The speech tokenizer's decoder loads as an
//! `engine.Vocoder` of the `Qwen3TTSCode2Wav` architecture.

use super::{Checkpoint, Constructor, Decoder};
use crate::tensor::{Device, HbmTensor};
use crate::{Context, Error, Shared};
use furiosa_opt_engine::decoder::{self, Served, Speaking};
use furiosa_opt_engine::speech::encoder as encoding;
use furiosa_opt_engine::speech::speaker as speaking;
use furiosa_opt_models::models::decoder::attention::qwen3_tts::speaker;
use furiosa_opt_models::models::decoder::attention::qwen3_tts::{self, codec};
use furiosa_opt_models::tensor::memory::Dtype;
use furiosa_opt_std::prelude::{AxisName, HbmBuffer};
use numpy::PyArrayMethods;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use std::io;
use std::sync::{Arc, PoisonError};

/// Adds the Qwen3-TTS classes to `module`.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ForConditionalGeneration>()?;
    module.add_class::<Speaker>()?;
    module.add_class::<SpeakerEncoder>()?;
    module.add_class::<SpeechEncoder>()
}

/// The Qwen3-TTS decoder classes by name, as the constructors of their objects.
pub(super) const CLASSES: &[(&str, Constructor)] = &[
    (
        "ForConditionalGeneration",
        Decoder::object::<ForConditionalGeneration>,
    ),
    ("Speaker", Decoder::object::<Speaker>),
];

/// HF `Qwen3TTSTalkerForConditionalGeneration`, loaded on a device: the talker's body, then its
/// greedy codec head over the codebook entries and the end of speech.
#[gen_stub_pyclass]
#[pyclass(frozen, extends = Decoder, module = "engine.qwen3_tts")]
#[derive(Default)]
pub(super) struct ForConditionalGeneration;

/// vLLM-Omni's `Qwen3TTSTalkerForConditionalGeneration`, loaded on a device: the talker, its
/// codec head and its code predictor, and the tables and projection its input rows read. A call
/// returns each kept row's FP32 scores of its frame's first entry, `[rows, 3328]`, the suppressed
/// control rows and the compiled rows past the 3072 lowest; every input row is an override the
/// caller builds with `text`, `codec`, `add` and `frame_rows`.
#[gen_stub_pyclass]
#[pyclass(frozen, extends = Decoder, module = "engine.qwen3_tts")]
#[derive(Default)]
pub(super) struct Speaker;

#[gen_stub_pymethods]
#[pymethods]
impl Speaker {
    /// The frames of the last call's kept rows whose first entries are `firsts`, one each in kept
    /// order: `[rows, 16]` I32 on the host, the other fifteen entries drawn from the code
    /// predictor's FP32 scores, the predictor fed the row's hidden values and the entries before.
    /// A draw is the first highest at `temperature` 0, else from the softmax over `temperature`
    /// of the `top_k` highest (all at 0) and of those the fewest whose mass reaches `top_p`; each
    /// row draws from its seed of `seeds`. A call's frames are given once.
    ///
    /// Raises:
    ///     ValueError: No call precedes, or `firsts` or `seeds` holds another count than its kept
    ///         rows.
    ///     RuntimeError: The device fails.
    #[pyo3(signature = (firsts, seeds, temperature = 0.0, top_k = 0, top_p = 1.0))]
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.int32]", imports = ("numpy", "numpy.typing")))]
    fn frames(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "collections.abc.Sequence[int]", imports = ("collections.abc",)))]
        firsts: Vec<i32>,
        #[gen_stub(override_type(type_repr = "collections.abc.Sequence[int]", imports = ("collections.abc",)))]
        seeds: Vec<u64>,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<Py<PyAny>, Error> {
        let decoder = slf.as_super().get();
        let speaking = speaking(&*decoder.served)?;
        let sampling = decoder::Sampling {
            temperature,
            top_k,
            top_p,
        };
        let frames = py.detach(|| {
            let mut shared = decoder
                .context
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let Context { device, runtime } = &mut *shared;
            runtime.block_on(speaking.frames(device, &firsts, sampling, &seeds))
        })?;
        let rows = frames.len();
        let flat: Vec<i32> = frames.into_iter().flatten().collect();
        let array = numpy::PyArray1::from_vec(py, flat).reshape([rows, codec::front::GROUPS])?;
        Ok(array.into_any().unbind())
    }

    /// The talker's input rows after `frames`, one a frame: its sixteen entries' table rows summed
    /// in FP32 and rounded once to BF16, then the BF16 text row of `texts` at its index added, as
    /// Transformers sums them; `[rows, 2048]` BF16. At most 128 frames.
    ///
    /// Raises:
    ///     ValueError: The counts differ or exceed 128, an entry lies outside its codebook, or a
    ///         text row is not one BF16 row of 2048 on this device.
    ///     RuntimeError: The device fails.
    fn frame_rows(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "collections.abc.Sequence[collections.abc.Sequence[int]]", imports = ("collections.abc",)))]
        frames: Vec<Vec<i32>>,
        #[gen_stub(override_type(type_repr = "collections.abc.Sequence[engine.HbmTensor]", imports = ("collections.abc",)))]
        texts: Vec<PyRef<'_, HbmTensor>>,
    ) -> Result<HbmTensor, Error> {
        let decoder = slf.as_super().get();
        let speaking = speaking(&*decoder.served)?;
        let frames = (frames.into_iter())
            .map(|frame| {
                <[i32; codec::front::GROUPS]>::try_from(frame).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "a frame holds sixteen entries")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let texts = (texts.iter())
            .map(|text| row(text, &decoder.context))
            .collect::<Result<Vec<_>, _>>()?;
        run(py, &decoder.context, async |device| {
            speaking.rows(device, &frames, &texts).await?.try_into()
        })
    }

    /// The talker's text rows of the text token IDs `ids`: `text_embedding`, then
    /// `text_projection`, `[rows, 2048]` BF16.
    ///
    /// Raises:
    ///     ValueError: An ID lies outside the text table.
    ///     RuntimeError: The device fails.
    fn text(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "collections.abc.Sequence[int]", imports = ("collections.abc",)))]
        ids: Vec<i32>,
    ) -> Result<HbmTensor, Error> {
        let decoder = slf.as_super().get();
        let speaking = speaking(&*decoder.served)?;
        run(py, &decoder.context, async |device| {
            speaking.text(device, &ids).await?.try_into()
        })
    }

    /// The talker's codec table rows of `ids`, which a prompt's control rows read: `[rows, 2048]`
    /// BF16.
    ///
    /// Raises:
    ///     ValueError: An ID lies outside the talker's 3072 rows.
    ///     RuntimeError: The device fails.
    fn codec(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "collections.abc.Sequence[int]", imports = ("collections.abc",)))]
        ids: Vec<i32>,
    ) -> Result<HbmTensor, Error> {
        let decoder = slf.as_super().get();
        let speaking = speaking(&*decoder.served)?;
        run(py, &decoder.context, async |device| {
            speaking.codec(device, &ids).await?.try_into()
        })
    }

    /// The rows of `pairs`, each the BF16 sum of its two rows, `None` zero, as Transformers adds a
    /// prompt's text and codec rows: `[rows, 2048]` BF16. A row is one BF16 row of 2048 on this
    /// device, as `text` and `codec` rows index.
    ///
    /// Raises:
    ///     ValueError: A row is not one BF16 row of 2048 on this device.
    ///     RuntimeError: The device fails.
    fn add(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "collections.abc.Sequence[tuple[engine.HbmTensor | None, engine.HbmTensor | None]]", imports = ("collections.abc",)))]
        pairs: Vec<(Option<PyRef<'_, HbmTensor>>, Option<PyRef<'_, HbmTensor>>)>,
    ) -> Result<HbmTensor, Error> {
        let decoder = slf.as_super().get();
        let speaking = speaking(&*decoder.served)?;
        let side = |tensor: &Option<PyRef<'_, HbmTensor>>| {
            tensor
                .as_ref()
                .map(|tensor| row(tensor, &decoder.context))
                .transpose()
        };
        let pairs = (pairs.iter())
            .map(|(left, right)| Ok((side(left)?, side(right)?)))
            .collect::<Result<Vec<_>, Error>>()?;
        run(py, &decoder.context, async |device| {
            speaking.add(device, &pairs).await?.try_into()
        })
    }
}

/// The speaker that `served` serves.
fn speaking(served: &dyn Served) -> PyResult<&Speaking> {
    let instance = served
        .as_any()
        .downcast_ref::<decoder::Instance<Speaking>>();
    Ok(instance
        .ok_or_else(|| PyTypeError::new_err("the decoder is not a Qwen3-TTS speaker"))?
        .decoder())
}

/// The storage of `tensor`, one BF16 row of the talker's width on the device of `context`.
fn row(tensor: &HbmTensor, context: &Shared) -> Result<HbmBuffer, Error> {
    let width = qwen3_tts::Text::SIZE;
    let one = matches!(tensor.shape[..], [w] | [1, w] if w == width);
    if !Arc::ptr_eq(&tensor.context, context) {
        return Err(PyValueError::new_err("a row lies on another device").into());
    }
    if !one || tensor.dtype != Dtype::BF16 {
        return Err(PyValueError::new_err("a row is one BF16 row of 2048").into());
    }
    Ok(tensor.buffer.alias(..width * 2).map_err(Error::from)?)
}

/// `work` on the device of `context`, without the GIL, as a tensor on that device.
fn run(
    py: Python<'_>,
    context: &Shared,
    work: impl AsyncFnOnce(&mut furiosa_opt_std::prelude::Device) -> io::Result<decoder::HbmTensor>
    + Send,
) -> Result<HbmTensor, Error> {
    let rows = py.detach(|| {
        let mut shared = context.lock().unwrap_or_else(PoisonError::into_inner);
        let Context { device, runtime } = &mut *shared;
        runtime.block_on(work(device))
    })?;
    Ok(HbmTensor {
        buffer: rows.buffer,
        shape: rows.shape,
        dtype: rows.dtype,
        context: context.clone(),
    })
}

/// The speaker encoder of a Qwen3-TTS Base checkpoint, the ECAPA-TDNN under `speaker_encoder`,
/// loaded on a device: its calls turn a reference clip's mel frames into the speaker embedding a
/// talker's prompt reads.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.qwen3_tts")]
pub(super) struct SpeakerEncoder {
    encoder: speaker::SpeakerEncoder,
    context: Shared,
}

#[gen_stub_pymethods]
#[pymethods]
impl SpeakerEncoder {
    /// The encoder under `speaker_encoder` of `checkpoint`, loaded strictly into HBM of `device`;
    /// the talker is not read.
    ///
    /// Raises:
    ///     ValueError: A tensor is missing, unexpected or malformed.
    ///     RuntimeError: The device fails.
    #[new]
    fn new(py: Python<'_>, checkpoint: &Checkpoint, device: &Device) -> Result<Self, Error> {
        let encoder = py.detach(|| {
            let mut context = device.0.lock().unwrap_or_else(PoisonError::into_inner);
            checkpoint
                .0
                .load::<speaker::SpeakerEncoder>(&mut context.device)
        })?;
        Ok(Self {
            encoder,
            context: device.0.clone(),
        })
    }

    /// The speaker embedding of a clip's `mels`, `[frames, 128]` FP32 log-mel frames of 24 kHz
    /// audio as qwen-tts computes them: `[1, 2048]` BF16 on the device, the talker's row of the
    /// speaker. The encoder runs in BF16 as Transformers runs it, its statistics in FP32.
    ///
    /// Raises:
    ///     ValueError: `mels` is not `[frames, 128]`, or holds fewer than 5 or more than 2048
    ///         frames, 21.8 s.
    ///     RuntimeError: The device fails.
    fn embed(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
        mels: numpy::PyReadonlyArray2<'_, f32>,
    ) -> Result<HbmTensor, Error> {
        let view = mels.as_array();
        if view.ncols() != speaker::Mel::SIZE {
            return Err(PyValueError::new_err("a mel frame holds 128 bins").into());
        }
        let values: Vec<f32> = view.iter().copied().collect();
        let embedding = py.detach(|| {
            let mut shared = self.context.lock().unwrap_or_else(PoisonError::into_inner);
            let Context { device, runtime } = &mut *shared;
            runtime.block_on(speaking::embed(device, &self.encoder, &values))
        })?;
        Ok(HbmTensor {
            buffer: embedding,
            shape: vec![1, speaker::Embedding::SIZE],
            dtype: Dtype::BF16,
            context: self.context.clone(),
        })
    }
}

/// The encoder of a Qwen3-TTS checkpoint's 12 Hz `speech_tokenizer`, loaded on a device: its calls
/// turn a reference clip's 24 kHz samples into the codec frames an in-context clone reads.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "engine.qwen3_tts")]
pub(super) struct SpeechEncoder {
    encoder: codec::encoder::Encoder,
    context: Shared,
}

#[gen_stub_pymethods]
#[pymethods]
impl SpeechEncoder {
    /// The encoder in the `speech_tokenizer` folder of `checkpoint`, loaded strictly into HBM of
    /// `device`; the tokenizer's decoder is not read.
    ///
    /// Raises:
    ///     ValueError: A tensor is missing, unexpected or malformed.
    ///     RuntimeError: The device fails.
    #[new]
    fn new(py: Python<'_>, checkpoint: &Checkpoint, device: &Device) -> Result<Self, Error> {
        let folder = checkpoint.0.folder("speech_tokenizer")?;
        let encoder = py.detach(|| {
            let mut context = device.0.lock().unwrap_or_else(PoisonError::into_inner);
            folder.load::<codec::encoder::Encoder>(&mut context.device)
        })?;
        Ok(Self {
            encoder,
            context: device.0.clone(),
        })
    }

    /// The codec frames of `samples`, a 24 kHz clip in [-1, 1] of up to 30.72 s: `[frames, 16]`
    /// I32, `ceil(samples / 1920)` frames, as Transformers' encode returns them. The device runs
    /// every layer, the codebooks' scores and the residuals; the host picks each code.
    ///
    /// Raises:
    ///     ValueError: `samples` is empty or longer than 737280 samples.
    ///     RuntimeError: The device fails.
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.int32]", imports = ("numpy", "numpy.typing")))]
    fn encode(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
        samples: numpy::PyReadonlyArray1<'_, f32>,
    ) -> Result<Py<PyAny>, Error> {
        let samples: Vec<f32> = samples.as_array().iter().copied().collect();
        let frames = py.detach(|| {
            let mut shared = self.context.lock().unwrap_or_else(PoisonError::into_inner);
            let Context { device, runtime } = &mut *shared;
            runtime.block_on(encoding::encode(device, &self.encoder, &samples))
        })?;
        let rows = frames.len();
        let flat: Vec<i32> = frames.into_iter().flatten().collect();
        let array = numpy::PyArray1::from_vec(py, flat).reshape([rows, codec::encoder::GROUPS])?;
        Ok(array.into_any().unbind())
    }
}
