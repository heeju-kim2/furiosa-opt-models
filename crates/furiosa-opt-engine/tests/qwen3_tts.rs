//! Qwen3-TTS through the engine's Rust API: the published checkpoint selects the talker, which loads
//! strictly under `talker` beside the modules it does not read; on a device, the talker fed
//! Transformers' input rows picks each frame's first codebook entry as Transformers does, and the
//! speech tokenizer's decoder turns the frames' codes into the samples Transformers decodes.

#![feature(adt_const_params, unsized_const_params)]

mod common;

use std::path::PathBuf;
use std::process::{Command, Stdio};

use furiosa_opt_engine::decoder::{
    Cache, HbmTensor, Instance, Keep, Output, Request, Rows, Sampling, Served, Speaking,
};
use furiosa_opt_engine::registry::Listed;
use furiosa_opt_engine::{Checkpoint, LoadOptions, Tiling};
use furiosa_opt_models::load::{Context, Load, Shapes};
use furiosa_opt_models::models::Implements;
use furiosa_opt_models::models::decoder::attention::qwen3_tts;
use furiosa_opt_models::models::decoder::form;
use furiosa_opt_models::tensor::cache;
use furiosa_opt_models::tensor::parallel::Parallel;
use furiosa_opt_std::prelude::{AxisName, Device, HbmBuffer, HostBufferSlice, M, Topology, bf16};
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

/// vLLM-Omni's code2wav architecture selects the 12 Hz speech tokenizer's decoder of the published
/// checkpoint, a speech decoder, from the config alone.
#[test]
fn selects_the_speech_decoder() {
    use furiosa_opt_engine::registry::Role;
    let config = LoadOptions::new()
        .architecture("Qwen3TTSCode2Wav")
        .unwrap()
        .parse(&QWEN3_TTS_1_7B.checkpoint())
        .unwrap();
    let class = config.class();
    assert_eq!(
        (class.family, class.name, config.size(), config.role()),
        ("qwen3_tts", "Code2Wav", "12Hz", Role::Vocoder)
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

/// A host pick: the first highest score greedily or with one candidate; else only among the
/// `top_k` highest, and only the highest once its mass alone reaches `top_p`; and each candidate
/// about as often as its softmax share over the temperature.
#[test]
fn samples_as_transformers() {
    let scores = [1.0, 3.0, 3.0, 2.0, f32::MIN, 0.5];
    assert_eq!(Sampling::GREEDY.pick(&scores, 0.7), 1);
    let sampled = |temperature, top_k, top_p| Sampling {
        temperature,
        top_k,
        top_p,
    };
    assert_eq!(sampled(0.9, 1, 1.0).pick(&scores, 0.99), 1);
    let draws = |sampling: Sampling| -> Vec<usize> {
        let mut counts = vec![0; scores.len()];
        for at in 0..10000 {
            counts[sampling.pick(&scores, (f64::from(at) + 0.5) / 10000.0)] += 1;
        }
        counts
    };
    let counts = draws(sampled(1.0, 3, 1.0));
    assert_eq!((counts[0], counts[4], counts[5]), (0, 0, 0));
    // e^3 : e^3 : e^2 of the three highest.
    let share = |count: usize| count as f64 / 10000.0;
    let total = 2.0 * 3f64.exp() + 2f64.exp();
    assert!((share(counts[1]) - 3f64.exp() / total).abs() < 0.01);
    assert!((share(counts[3]) - 2f64.exp() / total).abs() < 0.01);
    let counts = draws(sampled(0.01, 0, 0.4));
    assert_eq!(counts[1], 10000);
    let counts = draws(sampled(1.0, 0, 1.0));
    assert_eq!(counts[4], 0);
    assert!(counts[5] > 0);
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

/// The speaker, the talker and its code predictor on a device: the talker's input rows, the
/// prompt's and then each frame's, under `talker` in BF16.
type Speaker = qwen3_tts::Speaker<
    qwen3_tts::ForConditionalGeneration<
        qwen3_tts::Talker<"1.7B">,
        form::Affine,
        Parallel<1>,
        cache::Bf16,
        furiosa_opt_engine::Hbm,
    >,
    qwen3_tts::CodePredictor<
        qwen3_tts::Predictor<"1.7B">,
        form::Affine,
        Parallel<1>,
        cache::Bf16,
        furiosa_opt_engine::Hbm,
    >,
>;

/// The speaker on `device`, served adaptively.
fn speaker(device: &mut Device) -> Box<dyn Served> {
    let directory = snapshot(QWEN3_TTS_1_7B.hub(), &["*.json", "model.safetensors"]);
    let checkpoint = Checkpoint::new(directory, None).unwrap();
    let loaded: Speaker = Context::new(device)
        .field("", "talker", |cx| Load::load(cx, &checkpoint))
        .unwrap();
    let model = Listed::served(loaded, Tiling::Adaptive).decoder().unwrap();
    assert_eq!(model.output(), Output::Scores);
    model
}

/// The served speaker's own decoder.
fn speaking(model: &dyn Served) -> &Speaking {
    model
        .as_any()
        .downcast_ref::<Instance<Speaking>>()
        .unwrap()
        .decoder()
}

/// The first `bytes` bytes of `buffer` on the host, once the device wrote them.
async fn download(device: &mut Device, buffer: &HbmBuffer, bytes: usize) -> Vec<u8> {
    let mut host = vec![0; bytes];
    let first = buffer.alias(..bytes).unwrap();
    (first.to_host(&mut device.pdma).output(&mut host[..]).await).unwrap();
    host
}

/// The speaker fed Transformers' talker input rows one a call returns each kept row's FP32 scores
/// of the talker's rows, its suppressed control rows lowest, whose greedy pick is Transformers'
/// first entry but at its near-ties; given those picks, its frames' other entries, which the
/// predictor picks from the device talker's own hidden row and its own earlier picks, are
/// Transformers' up to a frame's first near-tie.
#[test]
#[ignore = "opens a device"]
fn speaks_each_frame_as_transformers() {
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
    let &[count, width] = rows.shape() else {
        panic!("input rows are a matrix");
    };
    let row = |at: usize| &rows.data()[at * width * 2..(at + 1) * width * 2];
    let codes = ids(&outputs, "codes");
    let groups = outputs.tensor("codes").unwrap().shape()[1];
    let frames = codes.len() / groups;
    assert_eq!(count, prefill + frames);

    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let model = speaker(&mut device);
    let speaking = speaking(&*model);
    let blocks: [u32; 3] = [2, 0, 1];
    runtime().block_on(async {
        let cache = Cache::new(&device, &model.limits().cache, 3).unwrap();
        let mut picked = Vec::new();
        for position in 0..count - 1 {
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
            let scores = model
                .forward(&mut device, &cache, std::slice::from_ref(&request), &[])
                .await
                .unwrap();
            assert_eq!(scores.shape, vec![1, qwen3_tts::Codec::SIZE]);
            if position + 1 >= prefill {
                let host = download(&mut device, &scores.buffer, qwen3_tts::Codec::SIZE * 4).await;
                let scores: Vec<f32> = (host.as_chunks::<4>().0.iter())
                    .map(|&b| f32::from_le_bytes(b))
                    .collect();
                // Greedy: the first of the highest.
                let first = Sampling::GREEDY.pick(&scores, 0.0);
                assert!(
                    first < 2048 || first == 2150,
                    "picked the suppressed row {first}"
                );
                let frames = speaking
                    .frames(&mut device, &[first as i32], Sampling::GREEDY, &[0])
                    .await
                    .unwrap();
                assert_eq!(frames.len(), 1);
                picked.extend(frames[0]);
            }
        }
        let first = |codes: &[i32]| codes.iter().step_by(groups).copied().collect::<Vec<_>>();
        let scores = outputs.tensor("first_scores").unwrap();
        let entries = scores.shape()[1];
        let score = |frame: usize, entry: i32| {
            let at = (frame * entries + entry as usize) * 2;
            bf16::from_bits(u16::from_le_bytes([
                scores.data()[at],
                scores.data()[at + 1],
            ]))
            .to_f32()
        };
        agree(&first(&picked), &first(&codes), score, 0.95, 0.01);
        // Up to its first difference a frame's picks are conditioned as Transformers' are, so that
        // difference is a near-tie of Transformers' scores; past it the frames are not compared.
        let rest = outputs.tensor("rest_scores").unwrap();
        let rest_score = |frame: usize, group: usize, entry: i32| {
            let at = ((frame * (groups - 1) + group - 1) * rest.shape()[2] + entry as usize) * 2;
            bf16::from_bits(u16::from_le_bytes([rest.data()[at], rest.data()[at + 1]])).to_f32()
        };
        let mut whole = 0;
        for frame in 0..frames {
            let (got, want) = (
                &picked[frame * groups..][..groups],
                &codes[frame * groups..][..groups],
            );
            match (0..groups).find(|&group| got[group] != want[group]) {
                None => whole += 1,
                Some(0) => {}
                Some(group) => {
                    let best = rest_score(frame, group, want[group]);
                    let below = (best - rest_score(frame, group, got[group])) / best.abs();
                    eprintln!(
                        "frame {frame}: entry {group} picked {} for {}, {below} below",
                        got[group], want[group]
                    );
                    assert!(
                        below <= 0.05,
                        "frame {frame}: entry {group} scored {below} below Transformers' pick"
                    );
                }
            }
        }
        eprintln!("{whole} of {frames} frames alike");
    });
}

/// The speaker builds Transformers' talker input rows on the device: the prompt's, each the BF16 sum
/// of a projected text row and a codec row or a text row alone, the projected rows close to
/// Transformers' and the sums of Transformers' rows its own; and each frame's, its sixteen entries'
/// rows summed in FP32 and rounded once, then its text row added, as Transformers sums them.
#[test]
#[ignore = "opens a device"]
fn builds_the_rows_as_transformers() {
    let (inputs, outputs, text) = (
        reference("inputs.safetensors"),
        reference("outputs.safetensors"),
        reference("text.safetensors"),
    );
    let (inputs, outputs, text) = (
        SafeTensors::deserialize(&inputs).unwrap(),
        SafeTensors::deserialize(&outputs).unwrap(),
        SafeTensors::deserialize(&text).unwrap(),
    );
    let prefill = ids(&inputs, "prefill")[0] as usize;
    let rows = inputs.tensor("talker_inputs").unwrap();
    let width = rows.shape()[1];
    let bytes = width * 2;
    let row = |at: usize| &rows.data()[at * bytes..(at + 1) * bytes];
    let codes = ids(&outputs, "codes");
    let frames: Vec<[i32; 16]> = codes.as_chunks::<16>().0.to_vec();
    let floats = |data: &[u8]| -> Vec<f32> {
        (data.as_chunks::<2>().0.iter())
            .map(|&b| bf16::from_bits(u16::from_le_bytes(b)).to_f32())
            .collect()
    };
    let text_ids = ids(&text, "text_ids");
    let trailing = text.tensor("trailing").unwrap();
    let (tts_bos, tts_pad) = (
        text.tensor("tts_bos").unwrap(),
        text.tensor("tts_pad").unwrap(),
    );
    // The config's tokens: the text's bos, eos and pad, then the codec's think, think bos, English,
    // think eos, pad and bos.
    let (bos, eos, pad) = (151672, 151673, 151671);
    let (think, think_bos, english, think_eos, codec_pad, codec_bos) =
        (2154, 2156, 2050, 2157, 2148, 2149);

    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let model = speaker(&mut device);
    let speaking = speaking(&*model);
    runtime().block_on(async {
        // The role's three rows, the text's first, its trailing rows and the text's specials.
        let length = text_ids.len();
        let mut wanted: Vec<i32> = text_ids[..4].to_vec();
        wanted.extend(&text_ids[4..length - 5]);
        wanted.extend([eos, bos, pad]);
        let projected = speaking.text(&mut device, &wanted).await.unwrap();
        assert_eq!(projected.len(), wanted.len());
        let host = download(&mut device, projected.buffer(), wanted.len() * bytes).await;
        let projected_row = |at: usize| &host[at * bytes..(at + 1) * bytes];
        let trailing_rows = trailing.shape()[0];
        let references: Vec<(&str, usize, &[u8])> = (0..3)
            .map(|at| ("role", at, row(at)))
            .chain((0..trailing_rows).map(|at| {
                (
                    "trailing",
                    4 + at,
                    &trailing.data()[at * bytes..(at + 1) * bytes],
                )
            }))
            .chain([("tts_bos", 4 + trailing_rows, tts_bos.data())])
            .chain([("tts_pad", 5 + trailing_rows, tts_pad.data())])
            .collect();
        for (name, at, want) in references {
            let (got, want) = (projected_row(at), want);
            let alike = (got.as_chunks::<2>().0.iter().zip(want.as_chunks::<2>().0))
                .filter(|(a, b)| a == b)
                .count();
            let similar = cosine(&floats(got), &floats(want));
            eprintln!("{name} row {at}: cosine {similar:.6}, {alike} of {width} alike");
            assert!(similar > 0.9999, "{name} row {at}: cosine {similar}");
        }

        // The prompt from Transformers' text rows and the device's codec rows: the role alone,
        // then the text pad and bos over the codec prefill, then the text's first over the codec
        // bos. Its speaker row, the pad over the clip's embedding, is not rebuilt.
        let codec = [think, think_bos, english, think_eos, codec_pad, codec_bos];
        let codec_rows = speaking.codec(&mut device, &codec).await.unwrap();
        let codec_row = |at: usize| {
            codec_rows
                .buffer()
                .alias(at * bytes..(at + 1) * bytes)
                .unwrap()
        };
        let mut pairs = Vec::new();
        for at in 0..3 {
            pairs.push((Some(upload(&mut device, row(at)).await), None));
        }
        for at in 0..4 {
            pairs.push((
                Some(upload(&mut device, tts_pad.data()).await),
                Some(codec_row(at)),
            ));
        }
        pairs.push((
            Some(upload(&mut device, tts_bos.data()).await),
            Some(codec_row(4)),
        ));
        pairs.push((
            Some(projected.buffer().alias(3 * bytes..4 * bytes).unwrap()),
            Some(codec_row(5)),
        ));
        let prompt = speaking.add(&mut device, &pairs).await.unwrap();
        let host = download(&mut device, prompt.buffer(), pairs.len() * bytes).await;
        let built = [0, 1, 2, 3, 4, 5, 6, 8, 9];
        assert_eq!(built.len(), pairs.len());
        assert_eq!(prefill, 10);
        for (at, &position) in built.iter().enumerate() {
            let (got, want) = (&host[at * bytes..(at + 1) * bytes], row(position));
            let similar = cosine(&floats(got), &floats(want));
            // The first text row is the device's projection, the others Transformers' own.
            if position == 9 {
                assert!(similar > 0.9999, "prompt row {position}: cosine {similar}");
            } else {
                assert!(
                    got == want,
                    "prompt row {position} differs, cosine {similar}"
                );
            }
        }

        // Each frame's row from Transformers' frame and text row.
        let count = rows.shape()[0] - prefill;
        let mut texts = Vec::new();
        for frame in 0..count {
            let text = match frame < trailing_rows {
                true => &trailing.data()[frame * bytes..(frame + 1) * bytes],
                false => tts_pad.data(),
            };
            texts.push(upload(&mut device, text).await);
        }
        let built = speaking
            .rows(&mut device, &frames[..count], &texts)
            .await
            .unwrap();
        let host = download(&mut device, built.buffer(), count * bytes).await;
        let differ: Vec<usize> = (0..count)
            .filter(|&frame| &host[frame * bytes..(frame + 1) * bytes] != row(prefill + frame))
            .collect();
        assert!(differ.is_empty(), "frame rows {differ:?} differ");
    });
}

// # The speaker encoder

/// The speaker encoder turns official qwen-tts's mel frames of an 8.5 s clip into its FP32 stages
/// and speaker embedding: each block's live rows, the aggregation's, the attentive mean and
/// standard deviation, then the embedding, as BF16 runs of Transformers' encoder come close to its
/// FP32 run.
#[test]
#[ignore = "opens a device"]
fn encodes_the_speaker_as_transformers() {
    use furiosa_opt_engine::speech::speaker;
    use furiosa_opt_models::models::decoder::attention::qwen3_tts::speaker::SpeakerEncoder;

    let file = reference("speaker.safetensors");
    let file = SafeTensors::deserialize(&file).unwrap();
    let (mels, shape) = stage(&file, "mels");
    let directory = snapshot(QWEN3_TTS_1_7B.hub(), &["*.json", "model.safetensors"]);
    let checkpoint = Checkpoint::new(directory, None).unwrap();
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let encoder: SpeakerEncoder = checkpoint.load(&mut device).unwrap();
    let bf16s = |bytes: Vec<u8>| -> Vec<f32> {
        (bytes.as_chunks::<2>().0.iter())
            .map(|&b| bf16::from_bits(u16::from_le_bytes(b)).to_f32())
            .collect()
    };
    let f32s = |bytes: Vec<u8>| -> Vec<f32> {
        (bytes.as_chunks::<4>().0.iter())
            .map(|&b| f32::from_le_bytes(b))
            .collect()
    };
    runtime().block_on(async {
        let encoded = speaker::encode(&mut device, &encoder, &mels).await.unwrap();
        let live = shape[0];
        assert_eq!(encoded.live, live);
        let similar = |name: &str, got: &[f32], floor: f64| {
            let (want, _) = stage(&file, &format!("{name}_f32"));
            let (bf16, _) = stage(&file, &format!("{name}_bf16"));
            let (device, transformers) = (cosine(got, &want), cosine(&bf16, &want));
            eprintln!("{name}: cosine {device:.6} to FP32, Transformers' BF16 {transformers:.6}");
            assert!(device >= floor, "{name}: cosine {device}");
        };
        for (at, block) in encoded.blocks.iter().enumerate() {
            let got = bf16s(download(&mut device, block, live * 512 * 2).await);
            similar(&format!("block{at}"), &got, 0.999);
        }
        let got = bf16s(download(&mut device, &encoded.aggregate, live * 1536 * 2).await);
        similar("mfa", &got, 0.999);
        let mut pooled = f32s(download(&mut device, &encoded.pooled[0], 1536 * 4).await);
        pooled.extend(f32s(
            download(&mut device, &encoded.pooled[1], 1536 * 4).await,
        ));
        similar("pooled", &pooled, 0.9995);
        let got = bf16s(download(&mut device, &encoded.embedding, 2048 * 2).await);
        similar("embedding", &got, 0.9999);
    });
}

// # The speech tokenizer's encoder

/// The speech tokenizer's encoder turns a 5 s clip into official qwen-tts's FP32 stages: SEANet's
/// frames, the transformer's and the downsampled codec frames close to them, and the codes of
/// most frames alike, every codebook's first code nearly always, as the residuals of the later
/// codebooks near-tie more often.
#[test]
#[ignore = "opens a device"]
fn encodes_the_reference_as_transformers() {
    use furiosa_opt_engine::speech::encoder;
    use furiosa_opt_models::models::decoder::attention::qwen3_tts::codec::encoder::Encoder;

    let file = reference("encoder.safetensors");
    let file = SafeTensors::deserialize(&file).unwrap();
    let (audio, _) = stage(&file, "audio");
    let directory = snapshot(QWEN3_TTS_1_7B.hub(), &["speech_tokenizer/*"]);
    let checkpoint = Checkpoint::new(directory.join("speech_tokenizer"), None).unwrap();
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let speech: Encoder = checkpoint.load(&mut device).unwrap();
    let bf16s = |bytes: Vec<u8>| -> Vec<f32> {
        (bytes.as_chunks::<2>().0.iter())
            .map(|&b| bf16::from_bits(u16::from_le_bytes(b)).to_f32())
            .collect()
    };
    runtime().block_on(async {
        let start = std::time::Instant::now();
        let encoded = encoder::stages(&mut device, &speech, &audio).await.unwrap();
        eprintln!("encoded in {:?}", start.elapsed());
        let f32s = |bytes: Vec<u8>| -> Vec<f32> {
            (bytes.as_chunks::<4>().0.iter())
                .map(|&b| f32::from_le_bytes(b))
                .collect()
        };
        for (index, buffer) in &encoded.layers {
            let (want, shape) = stage(&file, &format!("layer{index}"));
            let got = f32s(download(&mut device, buffer, shape[0] * shape[1] * 4).await);
            eprintln!("layer{index}: cosine {:.6}", cosine(&got, &want));
        }
        for (name, buffer) in [
            ("seanet", &encoded.seanet),
            ("transformer", &encoded.transformed),
            ("downsampled", &encoded.latents),
        ] {
            let (want, shape) = stage(&file, name);
            let got = bf16s(download(&mut device, buffer, shape[0] * shape[1] * 2).await);
            let similar = cosine(&got, &want);
            eprintln!("{name}: cosine {similar:.6}");
            assert!(similar > 0.999, "{name}: cosine {similar}");
        }
        let codes = ids(&file, "codes");
        let want: Vec<[i32; 16]> = codes.as_chunks::<16>().0.to_vec();
        assert_eq!(encoded.codes.len(), want.len());
        let mut alike = [0usize; 16];
        for (got, want) in encoded.codes.iter().zip(&want) {
            for group in 0..16 {
                alike[group] += usize::from(got[group] == want[group]);
            }
        }
        eprintln!("codes alike by codebook of {}: {alike:?}", want.len());
        assert!(alike[0] * 10 >= want.len() * 9, "first codebook: {alike:?}");
    });
}

// # The speech tokenizer's decoder

/// The FP32 values and shape of the tensor `name`.
fn stage(file: &SafeTensors<'_>, name: &str) -> (Vec<f32>, Vec<usize>) {
    let view = file.tensor(name).unwrap();
    assert_eq!(view.dtype(), Dtype::F32, "{name}");
    let values = (view.data().as_chunks::<4>().0.iter())
        .map(|&value| f32::from_le_bytes(value))
        .collect();
    (values, view.shape().to_vec())
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

/// The live rows of `got`, `width` channels a row, against the stage `name`: its rows and its
/// channels, the first of each of `got`'s rows. Returns the cosine of the whole stage.
fn compare(codec: &SafeTensors<'_>, name: &str, got: &[bf16], width: usize) -> f64 {
    let (want, shape) = stage(codec, name);
    let (rows, channels) = (shape[0], shape[1]);
    let got: Vec<f32> = (0..rows * channels)
        .map(|at| got[at / channels * width + at % channels].to_f32())
        .collect();
    let agreement = cosine(&got, &want);
    let worst = (got.iter().zip(&want))
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().map(|w| w.abs()).fold(0f32, f32::max);
    eprintln!("{name}: {rows} x {channels}, cosine {agreement:.6}, max error {worst} of {scale}");
    agreement
}

/// The speech decoder past its transformer on a device, fed Transformers' transformer rows, gives
/// each later stage's rows and the samples within a cosine of 0.99 of Transformers' FP32 ones.
#[test]
#[ignore = "opens a device"]
fn decodes_the_latents_as_transformers() {
    use furiosa_opt_engine::speech::qwen3_tts::{
        self as speech, First, Fourth, Rows1, Rows2, Rows3, Rows4, Second, Signal, Third,
    };
    use furiosa_opt_models::models::decoder::attention::qwen3_tts::codec::{
        FRAMES, Fifth, Frames, Latent, Latents, Samples, Twice, Waveform,
    };
    use furiosa_opt_std::prelude::{Pair, Symbol, m};

    let codec = reference("codec.safetensors");
    let codec = SafeTensors::deserialize(&codec).unwrap();
    let directory = snapshot(QWEN3_TTS_1_7B.hub(), &["speech_tokenizer/*"]);
    let checkpoint = Checkpoint::new(directory.join("speech_tokenizer"), None).unwrap();
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let waveform: Waveform = checkpoint.load(&mut device).unwrap();
    let (latent, shape) = stage(&codec, "pre_transformer");
    assert!(shape[0] <= FRAMES && shape[1] == Latent::SIZE);
    let mut rows = vec![0u8; FRAMES * Latent::SIZE * 2];
    for (at, value) in latent.iter().enumerate() {
        rows[at * 2..at * 2 + 2].copy_from_slice(&bf16::from_f32(*value).to_bits().to_le_bytes());
    }
    async fn host<R: AxisName, W: AxisName>(device: &mut Device, signal: &Signal<R, W>) -> Vec<bf16>
    where
        m![R, W]: M,
    {
        signal
            .to_host::<m![R, W]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec()
    }
    runtime().block_on(async {
        let framed = &waveform.framed;
        let x: Signal<Frames, Latent> = upload(&mut device, &rows).await.try_into().unwrap();
        let [first, second] = &waveform.upsample;
        let mut agreements = Vec::new();
        let x: Signal<Twice, Latent> = speech::upsample(&mut device, first, &framed.twice, &x)
            .await
            .unwrap();
        agreements.push(compare(
            &codec,
            "upsample0",
            &host(&mut device, &x).await,
            Latent::SIZE,
        ));
        let x: Signal<Latents, Latent> = speech::upsample(&mut device, second, &framed.latents, &x)
            .await
            .unwrap();
        agreements.push(compare(
            &codec,
            "upsample1",
            &host(&mut device, &x).await,
            Latent::SIZE,
        ));
        let x = speech::convolve(&mut device, &waveform, x).await.unwrap();
        agreements.push(compare(
            &codec,
            "decoder0",
            &host(&mut device, &x).await,
            First::SIZE,
        ));
        let x: Signal<Rows1, Second> = speech::block(
            &mut device,
            &waveform.block1,
            (
                (&framed.first, qwen3_tts::codec::LEAD),
                (&framed.second, qwen3_tts::codec::LEADS[0]),
            ),
            &x,
        )
        .await
        .unwrap();
        agreements.push(compare(
            &codec,
            "decoder1",
            &host(&mut device, &x).await,
            Second::SIZE,
        ));
        let x: Signal<Rows2, Third> = speech::block(
            &mut device,
            &waveform.block2,
            (
                (&framed.second, qwen3_tts::codec::LEADS[0]),
                (&framed.third, qwen3_tts::codec::LEADS[1]),
            ),
            &x,
        )
        .await
        .unwrap();
        agreements.push(compare(
            &codec,
            "decoder2",
            &host(&mut device, &x).await,
            Third::SIZE,
        ));
        // The later blocks read rows of two and of four samples, as their units' rows hold.
        let x: Signal<speech::Pairs, qwen3_tts::codec::Third2> = speech::recast(x).unwrap();
        let x: Signal<Rows3, Fourth> = speech::block(
            &mut device,
            &waveform.block3,
            (
                (&framed.third, qwen3_tts::codec::LEADS[1] / 2),
                (&framed.fourth, qwen3_tts::codec::LEADS[2]),
            ),
            &x,
        )
        .await
        .unwrap();
        agreements.push(compare(
            &codec,
            "decoder3",
            &host(&mut device, &x).await,
            Fourth::SIZE,
        ));
        let x: Signal<speech::Quads, qwen3_tts::codec::Fourth4> = speech::recast(x).unwrap();
        let x: Signal<Rows4, Fifth> = speech::block(
            &mut device,
            &waveform.block4,
            (
                (&framed.fourth, qwen3_tts::codec::LEADS[2] / 4),
                (&framed.fifth, qwen3_tts::codec::LEADS[3]),
            ),
            &x,
        )
        .await
        .unwrap();
        agreements.push(compare(
            &codec,
            "decoder4",
            &host(&mut device, &x).await,
            Fifth::SIZE,
        ));
        let wave = speech::output(&mut device, &waveform, x).await.unwrap();
        let samples = host(&mut device, &wave).await;
        agreements.push(compare(&codec, "wav", &samples, 1));
        assert_eq!(Samples::SIZE, 128);
        for agreement in agreements {
            assert!(agreement >= 0.99, "cosine {agreement}");
        }
    });
}

/// The whole speech decoder on a device turns the reference codes into each stage Transformers
/// computes, the samples within a cosine of 0.99 of Transformers' FP32 ones.
#[test]
#[ignore = "opens a device"]
fn decodes_the_codes_as_transformers() {
    use furiosa_opt_engine::speech::qwen3_tts::{self as speech, Quantized, Signal};
    use furiosa_opt_models::models::decoder::attention::qwen3_tts::codec::front::{GROUPS, Picks};
    use furiosa_opt_models::models::decoder::attention::qwen3_tts::codec::{Codec, Frames, Latent};
    use furiosa_opt_models::tensor::parallel::Chip;
    use furiosa_opt_std::prelude::{HbmTensor, Pair, Symbol, m};

    let codec_reference = reference("codec.safetensors");
    let codec_reference = SafeTensors::deserialize(&codec_reference).unwrap();
    let outputs = reference("outputs.safetensors");
    let outputs = SafeTensors::deserialize(&outputs).unwrap();
    let codes = outputs.tensor("codes").unwrap();
    assert_eq!((codes.dtype(), codes.shape()[1]), (Dtype::I32, GROUPS));
    let codes: Vec<[i32; GROUPS]> = (codes.data().as_chunks::<4>().0.iter())
        .map(|&value| i32::from_le_bytes(value))
        .collect::<Vec<_>>()
        .as_chunks::<GROUPS>()
        .0
        .to_vec();
    let offsets = speech::offsets(&codes).unwrap();
    let bytes: Vec<u8> = offsets
        .iter()
        .flat_map(|offset| offset.to_le_bytes())
        .collect();
    let directory = snapshot(QWEN3_TTS_1_7B.hub(), &["speech_tokenizer/*"]);
    let checkpoint = Checkpoint::new(directory.join("speech_tokenizer"), None).unwrap();
    let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let codec: Codec = checkpoint.load(&mut device).unwrap();
    async fn host<R: AxisName, W: AxisName>(device: &mut Device, signal: &Signal<R, W>) -> Vec<bf16>
    where
        m![R, W]: M,
    {
        signal
            .to_host::<m![R, W]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec()
    }
    runtime().block_on(async {
        let offsets: HbmTensor<i32, Chip, m![Picks]> =
            upload(&mut device, &bytes).await.try_into().unwrap();
        let mut agreements = Vec::new();
        let x: Signal<Frames, Quantized> = speech::quantize(&mut device, &codec.front, &offsets)
            .await
            .unwrap();
        let got = host(&mut device, &x).await;
        agreements.push(compare(
            &codec_reference,
            "quantized",
            &got,
            Quantized::SIZE,
        ));
        let x = speech::convolve_codes(&mut device, &codec.front, x)
            .await
            .unwrap();
        let got = host(&mut device, &x).await;
        agreements.push(compare(&codec_reference, "pre_conv", &got, Latent::SIZE));
        let x = speech::transform(&mut device, &codec.front, &x)
            .await
            .unwrap();
        let got = host(&mut device, &x).await;
        agreements.push(compare(
            &codec_reference,
            "pre_transformer",
            &got,
            Latent::SIZE,
        ));
        let wave = speech::decode(&mut device, &codec.waveform, &x)
            .await
            .unwrap();
        let samples = host(&mut device, &wave).await;
        agreements.push(compare(&codec_reference, "wav", &samples, 1));
        let wave = speech::synthesize(&mut device, &codec, &offsets)
            .await
            .unwrap();
        let again = host(&mut device, &wave).await;
        assert_eq!(
            again, samples,
            "the chained launch decodes the same samples"
        );
        for agreement in agreements {
            assert!(agreement >= 0.99, "cosine {agreement}");
        }
    });
}
