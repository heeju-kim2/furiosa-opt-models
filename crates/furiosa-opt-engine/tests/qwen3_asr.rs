//! Qwen3-ASR through the engine's Rust API: the published checkpoint selects the text model, which
//! loads strictly under `model` beside the audio tower it ignores, and for its encoder the tower;
//! on a device, the tower turns the clip's mel frames into Transformers' audio rows, and the text
//! model fed them at the prompt's placeholders transcribes the clip as Transformers does.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::path::PathBuf;
use std::process::{Command, Stdio};

use furiosa_opt_engine::decoder::{Cache, HbmTensor, Keep, Request, Rows};
use furiosa_opt_engine::{Checkpoint, LoadOptions};
use furiosa_opt_models::load::{Context, Load, Shapes};
use furiosa_opt_models::models::Implements;
use furiosa_opt_models::models::decoder::attention::qwen3;
use furiosa_opt_models::models::decoder::form;
use furiosa_opt_models::models::vision_language::qwen3_asr;
use furiosa_opt_models::models::vision_language::qwen3_asr::audio::Tower;
use furiosa_opt_models::tensor::cache;
use furiosa_opt_models::tensor::parallel::Parallel;
use furiosa_opt_std::prelude::{Device, HbmBuffer, HostBufferSlice, Topology, bf16};
use safetensors::{Dtype, SafeTensors};

use common::Published;

const QWEN3_ASR_1_7B: Published = Published::new(
    "Qwen/Qwen3-ASR-1.7B-hf",
    "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
);

/// The reference outputs of `QWEN3_ASR_1_7B`: the processor's prompt of the first 4 s of a speech
/// clip, the audio rows Transformers' tower and projector compute of it, and Transformers' greedy
/// transcript, "language English<asr_text>We have gone through a presidency that.<|im_end|>".
const REFERENCE: &str = "references/Qwen--Qwen3-ASR-1.7B-hf@bcd2b5b";

/// The prompt's `<|audio_pad|>` token, whose rows the audio rows replace.
const AUDIO: i32 = 151676;

/// The BF16 text model over the memory that checks shapes.
type Text = qwen3_asr::ForConditionalGeneration<
    qwen3::Size<"1.7B">,
    form::Affine,
    Parallel<1>,
    cache::Bf16,
    Shapes,
>;

// # The published checkpoint

/// The published config selects the 1.7B text model of the `qwen3_asr` family, in BF16.
#[test]
fn selects_the_text_model() {
    let config = LoadOptions::new()
        .parse(&QWEN3_ASR_1_7B.checkpoint())
        .unwrap();
    let class = config.class();
    assert_eq!(
        (class.family, class.name, config.size(), config.precision()),
        ("qwen3_asr", "ForConditionalGeneration", "1.7B", Dtype::BF16)
    );
}

/// The text model reads every name under `model.language_model`, ties its head to the token table
/// and ignores the audio tower and its projector.
#[test]
fn text_model_loads_strictly() {
    let checkpoint = QWEN3_ASR_1_7B.declared();
    let scope = Text::scope(&checkpoint).unwrap();
    assert_eq!(scope, "model");
    Context::new(None)
        .field("", scope, |cx| <Text as Load<_>>::load(cx, &checkpoint))
        .unwrap();
}

/// A load of the media encoder alone selects the audio tower and its projector, which read every
/// name under `model.audio_tower` and `model.multi_modal_projector` and ignore the text model.
#[test]
fn the_encoder_is_the_audio_tower() {
    let config = LoadOptions::new()
        .encoder()
        .parse(&QWEN3_ASR_1_7B.checkpoint())
        .unwrap();
    let class = config.class();
    assert_eq!((class.family, class.name), ("qwen3_asr", "Encoder"));
    let checkpoint = QWEN3_ASR_1_7B.declared().encoder();
    let scope = <Tower as Implements>::scope(&checkpoint).unwrap();
    assert_eq!(scope, "model");
}

// # On a device

/// The directory of the files of hub checkpoint `repo` at `revision` that `include` matches,
/// fetched into the hub cache once.
fn snapshot((repo, revision): (&str, &str), include: &[&str]) -> PathBuf {
    let mut command = Command::new("uvx");
    command.args(["--from", "huggingface-hub", "hf", "download"]);
    command.args([repo, "--revision", revision, "--quiet"]);
    for pattern in include {
        command.args(["--include", pattern]);
    }
    let output = command
        .stderr(Stdio::inherit())
        .output()
        .expect("uvx fetches the test checkpoint");
    assert!(output.status.success(), "the {repo} download failed");
    String::from_utf8(output.stdout).unwrap().trim().into()
}

/// The bytes of the reference file `name`, where `FURIOSA_OPT_REFERENCES` holds the release.
fn reference(name: &str) -> Vec<u8> {
    let root = std::env::var("FURIOSA_OPT_REFERENCES").expect("FURIOSA_OPT_REFERENCES is set");
    let path = PathBuf::from(root).join(REFERENCE).join(name);
    std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// The I32 values of the tensor `name` of a reference file.
fn ids(file: &SafeTensors<'_>, name: &str) -> Vec<i32> {
    let view = file.tensor(name).unwrap();
    assert_eq!(view.dtype(), Dtype::I32, "{name}");
    let (values, []) = view.data().as_chunks::<4>() else {
        panic!("{name} holds whole I32 values");
    };
    values
        .iter()
        .map(|&value| i32::from_le_bytes(value))
        .collect()
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// `bytes` in HBM of `device`.
async fn upload(device: &mut Device, bytes: &[u8]) -> HbmBuffer {
    HostBufferSlice::from(bytes)
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}

/// The I32 token of a one-row output, read back once its call completed.
async fn token(device: &mut Device, tokens: HbmTensor) -> i32 {
    let mut host = [0; 4];
    tokens
        .buffer
        .to_host(&mut device.pdma)
        .output(&mut host[..])
        .await
        .unwrap();
    i32::from_le_bytes(host)
}

/// The text model fed the prompt one row a call through a scattered block table, each audio
/// placeholder's row replaced by Transformers' audio row, continues it with Transformers' greedy
/// transcript.
#[test]
#[ignore = "opens a device"]
fn transcribes_the_clip_as_transformers() {
    let (inputs, outputs) = (
        reference("inputs.safetensors"),
        reference("outputs.safetensors"),
    );
    let (inputs, outputs) = (
        SafeTensors::deserialize(&inputs).unwrap(),
        SafeTensors::deserialize(&outputs).unwrap(),
    );
    let prompt = ids(&inputs, "prompt");
    let continuation = ids(&outputs, "continuation");
    let audio = outputs.tensor("audio_rows").unwrap();
    assert_eq!(audio.dtype(), Dtype::BF16);
    let &[count, width] = audio.shape() else {
        panic!("audio rows are a matrix");
    };
    let first = prompt.iter().position(|&token| token == AUDIO).unwrap();
    assert!(prompt[first..first + count].iter().all(|&t| t == AUDIO));
    let row = |at: usize| &audio.data()[at * width * 2..(at + 1) * width * 2];

    let directory = snapshot(QWEN3_ASR_1_7B.hub(), &["*.json", "*.safetensors"]);
    let checkpoint = Checkpoint::new(directory, None).unwrap();
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let model = LoadOptions::new()
        .load(&mut device, &checkpoint)
        .unwrap()
        .decoder()
        .unwrap();
    // 96 positions over six of the seven blocks, out of order.
    let blocks: [u32; 6] = [2, 0, 5, 3, 6, 1];
    runtime().block_on(async {
        let cache = Cache::new(&device, &model.limits().cache, 7).unwrap();
        let (mut selected, mut transcript) = (0, Vec::new());
        for position in 0..prompt.len() + continuation.len() - 1 {
            let input = prompt.get(position).copied().unwrap_or(selected);
            let ids = upload(&mut device, &input.to_le_bytes()).await;
            let overrides = match position.checked_sub(first) {
                Some(at) if at < count => vec![(0, upload(&mut device, row(at)).await)],
                _ => Vec::new(),
            };
            let request = Request {
                rows: vec![blocks.to_vec()],
                cached: position,
                tokens: Rows::new(&ids, 0..1).unwrap(),
                keep: Keep::Last,
                adapter: None,
                positions: None,
                mask: None,
                overrides,
            };
            let tokens = model
                .forward(&mut device, &cache, std::slice::from_ref(&request), &[])
                .await
                .unwrap();
            selected = token(&mut device, tokens).await;
            if position + 1 >= prompt.len() {
                transcript.push(selected);
            }
        }
        assert_eq!(transcript, continuation);
    });
}

/// The cosine of `got` and `want`.
fn cosine(got: &[f32], want: &[f32]) -> f64 {
    let dot = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum()
    };
    dot(got, want) / (dot(got, got) * dot(want, want)).sqrt()
}

/// The BF16 values of `bytes`.
fn values(bytes: &[u8]) -> Vec<f32> {
    (bytes.as_chunks::<2>().0.iter())
        .map(|&value| bf16::from_bits(u16::from_le_bytes(value)).to_f32())
        .collect()
}

/// The tower's convolutions, flattening and positions on a device turn the clip's mel frames into
/// the rows Transformers' do, each within a cosine of 0.999.
#[test]
#[ignore = "opens a device"]
fn embeds_the_clip_as_transformers() {
    use furiosa_opt_engine::vision_language::qwen3_asr::{self as engine, Embedded, Frames, Times};
    use furiosa_opt_models::models::vision_language::qwen3_asr::audio::{Eight, Frontend, Width};
    use furiosa_opt_std::prelude::{Pair, Symbol, m};
    let inputs = reference("inputs.safetensors");
    let inputs = SafeTensors::deserialize(&inputs).unwrap();
    let encoder = reference("encoder.safetensors");
    let encoder = SafeTensors::deserialize(&encoder).unwrap();
    let rows = mel_frames(&inputs);
    let directory = snapshot(QWEN3_ASR_1_7B.hub(), &["*.json", "*.safetensors"]);
    let checkpoint = Checkpoint::new(directory, None).unwrap();
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let frontend: Frontend = Context::new(&mut device)
        .field("", "model.audio_tower", |cx| Load::load(cx, &checkpoint))
        .unwrap();
    let want = encoder.tensor("embedded").unwrap();
    let (chunks, times, width) = (want.shape()[0], want.shape()[1], want.shape()[2]);
    let want = values(want.data());
    runtime().block_on(async {
        let frames: Frames<Eight> = upload(&mut device, &rows).await.try_into().unwrap();
        let offsets = &frontend.offsets.2;
        let embedded: Embedded = engine::embed(&frontend, &mut device, &frames, offsets)
            .await
            .unwrap();
        let got = embedded
            .to_host::<m![Times, Width]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let got: Vec<f32> = got.iter().map(|value| value.to_f32()).collect();
        for row in 0..chunks * times {
            let (got, want) = (&got[row * width..][..width], &want[row * width..][..width]);
            let agreement = cosine(got, want);
            assert!(agreement >= 0.999, "row {row}: cosine {agreement}");
        }
    });
}

/// The reference prompt's mel frames as a launch holds them: frame-major BF16 rows of 128
/// features, the launch's chunks past the clip's and its last frame all zeros.
fn mel_frames(inputs: &SafeTensors<'_>) -> Vec<u8> {
    use furiosa_opt_models::models::vision_language::qwen3_asr::audio::{CHUNKS, FRAMES};
    let mel = inputs.tensor("input_features").unwrap();
    let (bins, frames) = (mel.shape()[0], mel.shape()[1]);
    let mel: Vec<f32> = (mel.data().as_chunks::<4>().0.iter())
        .map(|&value| f32::from_le_bytes(value))
        .collect();
    let mut rows = vec![0u8; (CHUNKS * FRAMES + 1) * bins * 2];
    for frame in 0..frames {
        for bin in 0..bins {
            let value = bf16::from_f32(mel[bin * frames + frame])
                .to_bits()
                .to_le_bytes();
            rows[(frame * bins + bin) * 2..][..2].copy_from_slice(&value);
        }
    }
    rows
}

/// The checkpoint's media encoder, the tower and projector, turns the clip's mel frames on a device
/// into audio rows within a cosine of 0.998 of Transformers', and the text model fed them
/// transcribes the clip as Transformers does: the whole of Qwen3-ASR on the device.
#[test]
#[ignore = "opens a device"]
fn transcribes_the_clip_from_its_mel_frames() {
    use furiosa_opt_engine::registry::Model;
    use furiosa_opt_engine::vision_language::Audio;
    let (inputs, outputs) = (
        reference("inputs.safetensors"),
        reference("outputs.safetensors"),
    );
    let (inputs, outputs) = (
        SafeTensors::deserialize(&inputs).unwrap(),
        SafeTensors::deserialize(&outputs).unwrap(),
    );
    let prompt = ids(&inputs, "prompt");
    let continuation = ids(&outputs, "continuation");
    let want = outputs.tensor("audio_rows").unwrap();
    let (count, width) = (want.shape()[0], want.shape()[1]);
    let want = values(want.data());
    let first = prompt.iter().position(|&token| token == AUDIO).unwrap();
    let directory = snapshot(QWEN3_ASR_1_7B.hub(), &["*.json", "*.safetensors"]);
    let checkpoint = Checkpoint::new(directory, None).unwrap();
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let encoder = match LoadOptions::new()
        .encoder()
        .load(&mut device, &checkpoint)
        .unwrap()
    {
        Model::Encoder(encoder) => encoder,
        Model::Decoder(_) => panic!("the encoder part loads a media encoder"),
    };
    let model = LoadOptions::new()
        .load(&mut device, &checkpoint)
        .unwrap()
        .decoder()
        .unwrap();
    let blocks: [u32; 6] = [2, 0, 5, 3, 6, 1];
    runtime().block_on(async {
        let frames = inputs.tensor("input_features").unwrap().shape()[1];
        let mel = mel_frames(&inputs);
        let clip = Audio::new(upload(&mut device, &mel[..frames * 128 * 2]).await, frames);
        let embedded = (encoder.audio(&mut device, std::slice::from_ref(&clip)))
            .await
            .unwrap();
        let [embedded] = &embedded[..] else {
            panic!("one clip embeds as one tensor");
        };
        assert_eq!(embedded.shape, [count, width]);
        let rows = embedded.buffer.alias(..).unwrap();
        let mut host = vec![0u8; rows.size()];
        rows.to_host(&mut device.pdma)
            .output(&mut host[..])
            .await
            .unwrap();
        let got = values(&host);
        let agreements: Vec<f64> = (0..count)
            .map(|row| cosine(&got[row * width..][..width], &want[row * width..][..width]))
            .collect();
        let least = agreements.iter().copied().fold(1.0, f64::min);
        eprintln!("least cosine of the audio rows: {least}");
        assert!(least >= 0.998, "{agreements:?}");
        let cache = Cache::new(&device, &model.limits().cache, 7).unwrap();
        let (mut selected, mut transcript) = (0, Vec::new());
        let row = width * 2;
        for position in 0..prompt.len() + continuation.len() - 1 {
            let input = prompt.get(position).copied().unwrap_or(selected);
            let ids = upload(&mut device, &input.to_le_bytes()).await;
            let overrides = match position.checked_sub(first) {
                Some(at) if at < count => vec![(0, rows.alias(at * row..(at + 1) * row).unwrap())],
                _ => Vec::new(),
            };
            let request = Request {
                rows: vec![blocks.to_vec()],
                cached: position,
                tokens: Rows::new(&ids, 0..1).unwrap(),
                keep: Keep::Last,
                adapter: None,
                positions: None,
                mask: None,
                overrides,
            };
            let tokens = model
                .forward(&mut device, &cache, std::slice::from_ref(&request), &[])
                .await
                .unwrap();
            selected = token(&mut device, tokens).await;
            if position + 1 >= prompt.len() {
                transcript.push(selected);
            }
        }
        assert_eq!(transcript, continuation);
    });
}
