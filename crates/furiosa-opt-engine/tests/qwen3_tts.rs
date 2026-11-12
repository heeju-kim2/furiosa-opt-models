//! Qwen3-TTS through the engine's Rust API: the published checkpoint selects the talker, which loads
//! strictly under `talker` beside the modules it does not read; on a device, the talker fed
//! Transformers' input rows picks each frame's first codebook entry as Transformers does.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::path::PathBuf;
use std::process::{Command, Stdio};

use furiosa_opt_engine::decoder::{Cache, HbmTensor, Keep, Request, Rows};
use furiosa_opt_engine::registry::Listed;
use furiosa_opt_engine::{Checkpoint, LoadOptions, Tiling};
use furiosa_opt_models::load::{Context, Load, Shapes};
use furiosa_opt_models::models::Implements;
use furiosa_opt_models::models::decoder::attention::qwen3_tts;
use furiosa_opt_models::models::decoder::form;
use furiosa_opt_models::tensor::cache;
use furiosa_opt_models::tensor::parallel::Parallel;
use furiosa_opt_std::prelude::{Device, HbmBuffer, HostBufferSlice, Topology, bf16};
use safetensors::{Dtype, SafeTensors};

use common::Published;

const QWEN3_TTS_1_7B: Published = Published::new(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "fd4b254389122332181a7c3db7f27e918eec64e3",
);

/// The reference outputs of `QWEN3_TTS_1_7B`: Transformers' greedy voice clone of "Hello, this is a
/// test." in English from a clip's speaker embedding, its talker's input rows, the prompt's and
/// then one a frame, each frame's sixteen codebook entries and its BF16 scores of the first.
const REFERENCE: &str = "references/Qwen--Qwen3-TTS-12Hz-1.7B-Base@fd4b254";

/// The codec pad, the token of a row whose input row the caller gives.
const PAD: i32 = 2148;

/// The BF16 code predictor over the memory that checks shapes.
type Predictor = qwen3_tts::CodePredictor<
    qwen3_tts::Predictor<"1.7B">,
    form::Affine,
    Parallel<1>,
    cache::Bf16,
    Shapes,
>;

/// The BF16 talker over the memory that checks shapes.
type Talker = qwen3_tts::ForConditionalGeneration<
    qwen3_tts::Talker<"1.7B">,
    form::Affine,
    Parallel<1>,
    cache::Bf16,
    Shapes,
>;

// # The published checkpoint

/// The published config selects the 1.7B talker of the `qwen3_tts` family, in BF16.
#[test]
fn selects_the_talker() {
    let config = LoadOptions::new()
        .parse(&QWEN3_TTS_1_7B.checkpoint())
        .unwrap();
    let class = config.class();
    assert_eq!(
        (class.family, class.name, config.size(), config.precision()),
        ("qwen3_tts", "ForConditionalGeneration", "1.7B", Dtype::BF16)
    );
}

/// The talker reads its body and codec head under `talker` and ignores its text table and
/// projection and its code predictor; the speaker encoder lies outside its scope.
#[test]
fn talker_loads_strictly() {
    let checkpoint = QWEN3_TTS_1_7B.declared();
    let scope = Talker::scope(&checkpoint).unwrap();
    assert_eq!(scope, "talker");
    Context::new(None)
        .field("", scope, |cx| <Talker as Load<_>>::load(cx, &checkpoint))
        .unwrap();
}

/// The code predictor reads the talker's codec table and its own projection, body, tables and
/// heads under `talker`, its body's token table none the checkpoint stores.
#[test]
fn code_predictor_loads_strictly() {
    let checkpoint = QWEN3_TTS_1_7B.declared();
    let scope = Predictor::scope(&checkpoint).unwrap();
    assert_eq!(scope, "talker");
    Context::new(None)
        .field("", scope, |cx| {
            <Predictor as Load<_>>::load(cx, &checkpoint)
        })
        .unwrap();
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

/// The talker fed Transformers' input rows one a call through a scattered block table picks, at
/// the prompt's last row and at each frame's but the last, the first codebook entry Transformers
/// picks, or one its BF16 scores tie with.
#[test]
#[ignore = "opens a device"]
fn picks_each_frame_as_transformers() {
    let (inputs, outputs) = (
        reference("inputs.safetensors"),
        reference("outputs.safetensors"),
    );
    let (inputs, outputs) = (
        SafeTensors::deserialize(&inputs).unwrap(),
        SafeTensors::deserialize(&outputs).unwrap(),
    );
    let prefill = ids(&inputs, "prefill")[0] as usize;
    let rows = inputs.tensor("talker_inputs").unwrap();
    assert_eq!(rows.dtype(), Dtype::BF16);
    let &[count, width] = rows.shape() else {
        panic!("input rows are a matrix");
    };
    let row = |at: usize| &rows.data()[at * width * 2..(at + 1) * width * 2];
    let codes = outputs.tensor("codes").unwrap();
    let groups = codes.shape()[1];
    let first: Vec<i32> = ids(&outputs, "codes").into_iter().step_by(groups).collect();
    // The last row picks a frame whose other entries Transformers' generation never predicted.
    assert_eq!(count, prefill + first.len());

    let directory = snapshot(QWEN3_TTS_1_7B.hub(), &["*.json", "model.safetensors"]);
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
        let mut picked = Vec::new();
        for position in 0..count {
            let ids = upload(&mut device, &PAD.to_le_bytes()).await;
            let request = Request {
                rows: vec![blocks.to_vec()],
                cached: position,
                tokens: Rows::new(&ids, 0..1).unwrap(),
                keep: Keep::Last,
                adapter: None,
                positions: None,
                mask: None,
                overrides: vec![(0, upload(&mut device, row(position)).await)],
            };
            let tokens = model
                .forward(&mut device, &cache, std::slice::from_ref(&request), &[])
                .await
                .unwrap();
            let selected = token(&mut device, tokens).await;
            if position + 1 >= prefill {
                picked.push(selected);
            }
        }
        // Transformers rounds its scores to BF16 and picks the first of a tie, where the device
        // picks the larger of its FP32 scores: a pick may differ only where Transformers' tie.
        let scores = outputs.tensor("first_scores").unwrap();
        let rows = scores.shape()[1];
        let score = |frame: usize, entry: i32| {
            let at = (frame * rows + entry as usize) * 2;
            u16::from_le_bytes([scores.data()[at], scores.data()[at + 1]])
        };
        let differ: Vec<usize> = (0..first.len())
            .filter(|&f| picked[f] != first[f])
            .collect();
        for &frame in &differ {
            let (device, transformers) = (picked[frame], first[frame]);
            assert_eq!(
                score(frame, device),
                score(frame, transformers),
                "frame {frame}: picked {device}, Transformers {transformers}"
            );
        }
        assert!(differ.len() <= 1, "frames {differ:?} differ");
    });
}

/// The picks of `picked` agree with Transformers' `expected`, whose BF16 scores of row `at` are
/// `score(at, entry)`: all but `1 - exact` of them alike, and each other one scored by Transformers
/// within `close` of its best, as a flat row's near-ties differ between the device's FP32 sums and
/// Transformers' BF16 rounding.
fn agree(
    picked: &[i32],
    expected: &[i32],
    score: impl Fn(usize, i32) -> f32,
    exact: f64,
    close: f32,
) {
    assert_eq!(picked.len(), expected.len());
    let differ: Vec<(usize, i32, i32, f32)> = (0..expected.len())
        .filter(|&at| picked[at] != expected[at])
        .map(|at| {
            let (best, device) = (score(at, expected[at]), score(at, picked[at]));
            (at, picked[at], expected[at], (best - device) / best.abs())
        })
        .collect();
    let alike = 1.0 - differ.len() as f64 / expected.len() as f64;
    assert!(alike >= exact, "{alike} of the picks alike: {differ:?}");
    for &(at, device, transformers, below) in &differ {
        assert!(
            below <= close,
            "row {at}: picked {device}, scored {below} below Transformers' {transformers}"
        );
    }
}

/// The code predictor, each frame a sequence of its own through a scattered block table, fed the
/// talker's hidden row of the frame and then Transformers' entries one a call, picks most later
/// codebooks' entries as Transformers does, and each other one among Transformers' near-ties.
#[test]
#[ignore = "opens a device"]
fn predicts_each_codebook_as_transformers() {
    let outputs = reference("outputs.safetensors");
    let outputs = SafeTensors::deserialize(&outputs).unwrap();
    let inputs = reference("inputs.safetensors");
    let inputs = SafeTensors::deserialize(&inputs).unwrap();
    let prefill = ids(&inputs, "prefill")[0] as usize;
    let hidden = outputs.tensor("talker_hidden").unwrap();
    let width = hidden.shape()[1];
    let talker = |at: usize| &hidden.data()[at * width * 2..(at + 1) * width * 2];
    let groups = outputs.tensor("codes").unwrap().shape()[1];
    let codes = ids(&outputs, "codes");
    let scores = outputs.tensor("rest_scores").unwrap();
    let entries = scores.shape()[2];

    let directory = snapshot(QWEN3_TTS_1_7B.hub(), &["*.json", "model.safetensors"]);
    let checkpoint = Checkpoint::new(directory, None).unwrap();
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    type Loaded = qwen3_tts::CodePredictor<
        qwen3_tts::Predictor<"1.7B">,
        form::Affine,
        Parallel<1>,
        cache::Bf16,
        furiosa_opt_engine::Hbm,
    >;
    let loaded: Loaded = Context::new(&mut device)
        .field("", "talker", |cx| Load::load(cx, &checkpoint))
        .unwrap();
    let model = Listed::served(loaded, Tiling::Adaptive).decoder().unwrap();
    let blocks: [u32; 2] = [1, 0];
    runtime().block_on(async {
        let cache = Cache::new(&device, &model.limits().cache, 2).unwrap();
        let (mut picked, mut expected) = (Vec::new(), Vec::new());
        for frame in 0..codes.len() / groups {
            let entry = |group: usize| codes[frame * groups + group];
            for position in 1..groups {
                let tokens: Vec<i32> = match position {
                    1 => vec![PAD, entry(0)],
                    _ => vec![entry(position - 1)],
                };
                let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
                let ids = upload(&mut device, &bytes).await;
                let overrides = match position {
                    1 => vec![(0, upload(&mut device, talker(prefill - 1 + frame)).await)],
                    _ => Vec::new(),
                };
                let request = Request {
                    rows: vec![blocks.to_vec()],
                    cached: position + 1 - tokens.len(),
                    tokens: Rows::new(&ids, 0..tokens.len()).unwrap(),
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
                picked.push(token(&mut device, tokens).await);
                expected.push(entry(position));
            }
        }
        let score = |at: usize, entry: i32| {
            let at = (at * entries + entry as usize) * 2;
            bf16::from_bits(u16::from_le_bytes([
                scores.data()[at],
                scores.data()[at + 1],
            ]))
            .to_f32()
        };
        agree(&picked, &expected, score, 0.85, 0.04);
    });
}
