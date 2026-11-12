//! Qwen3-ASR through the engine's Rust API: the published checkpoint selects the text model, which
//! loads strictly under `model` beside the audio tower it ignores; on a device, the text model fed
//! Transformers' audio rows at the prompt's placeholders transcribes the clip as Transformers does.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::path::PathBuf;
use std::process::{Command, Stdio};

use furiosa_opt_engine::decoder::{Cache, HbmTensor, Keep, Request, Rows};
use furiosa_opt_engine::{Checkpoint, LoadError, LoadOptions};
use furiosa_opt_models::load::{Context, Load, Shapes};
use furiosa_opt_models::models::Implements;
use furiosa_opt_models::models::decoder::attention::qwen3;
use furiosa_opt_models::models::decoder::form;
use furiosa_opt_models::models::vision_language::qwen3_asr;
use furiosa_opt_models::tensor::cache;
use furiosa_opt_models::tensor::parallel::Parallel;
use furiosa_opt_std::prelude::{Device, HbmBuffer, HostBufferSlice, Topology};
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

/// A load of the media encoder alone is refused: no type serves the audio tower yet.
#[test]
fn the_audio_tower_is_not_served_yet() {
    let encoder = LoadOptions::new()
        .encoder()
        .parse(&QWEN3_ASR_1_7B.checkpoint());
    assert!(
        matches!(&encoder, Err(LoadError::Unsupported { name, .. }) if name == "encoder"),
        "{:?}",
        encoder.map(|config| config.size())
    );
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
