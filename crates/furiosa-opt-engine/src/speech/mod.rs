//! Speech models on the device: a talker's codec frames into waveforms, and a reference clip into
//! the speaker embedding and the codec frames a talker's prompt reads.

pub mod encoder;
pub mod qwen3_tts;
pub mod speaker;

use furiosa_opt_std::prelude::Device;
use std::future::Future;
use std::io;
use std::pin::Pin;

/// A speech decoder a framework serves: codec frames, an entry of each codebook a frame, into
/// waveform samples.
pub trait Vocoder: Send + Sync {
    /// The codebooks of a frame, and the entries of each.
    fn groups(&self) -> usize;
    fn entries(&self) -> usize;

    /// The samples a frame decodes into, and their rate in hertz.
    fn samples(&self) -> usize;
    fn rate(&self) -> usize;

    /// The samples of `codes`, [`Self::groups`] entries a frame, frame-major, as FP32 at
    /// [`Self::rate`]; refuses a partial frame or an entry outside its codebook before any launch.
    fn decode<'a>(
        &'a self,
        device: &'a mut Device,
        codes: &'a [i32],
    ) -> Pin<Box<dyn Future<Output = io::Result<Vec<f32>>> + Send + 'a>>;
}
