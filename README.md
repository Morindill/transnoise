# transnoise

Streaming speech denoising, end to end: data pipeline, evaluation, model training (Keras 3 + TensorFlow), a hand-written dependency-free C++ inference core, and a JUCE VST3 plugin.

- **Model**: 48 kHz, STFT 960/480 (20 ms window, 10 ms hop), a causal sliding-window Transformer in the spirit of DeepFilterNet3. It predicts 32 ERB-band gains plus a low-frequency deep filter. Total algorithmic latency is about 20-30 ms.
- **Deployment**: the C++ core (`cpp/`, no third-party dependencies) is checked layer by layer against a numpy reference, which in turn is checked against the Keras model.
- **Plugin**: mono/stereo VST3 with Mix and Bypass parameters, 48 kHz only.

| Part | Location | Where it runs |
|---|---|---|
| Data pipeline, evaluation | `src/transnoise/{data_pipeline,eval}` | WSL, main uv environment |
| Model, training | `src/transnoise/{model,training}`, `train/` | WSL, separate uv environment in `train/` |
| Export, numpy reference | `src/transnoise/deploy/` | export in the train environment, the rest in the main environment |
| C++ inference core | `cpp/` | WSL, g++ + make |
| VST3 plugin | `plugin/` | Windows, MSVC + CMake |
| Baseline models for comparison | `models/{rnnoise,dfn3,mossformer2}` | one-off inference tools, each with its own environment |

> Messages printed by the command-line tools (progress, check results, error messages) are in Chinese; code comments and this document are in English.

---

## 0. Quick map of the whole workflow

```
setup  ->  data  ->  test sets  ->  (checks)  ->  train  ->  infer + score  ->  export + C++  ->  plugin
 §1        §2        §3             §4            §5         §6                §7              §8
```

All commands are run from the repository root (`~/transnoise` in WSL) unless a `cd` is shown. Python is only ever used through `uv` environments; never `pip install` globally.

---

## 1. Setup

### 1.1 Requirements

| What | Needed for | Notes |
|---|---|---|
| Linux (WSL2 Ubuntu works) | everything except the plugin | the commands below assume a Linux shell |
| [uv](https://docs.astral.sh/uv/) | all Python environments | Python >= 3.12 is picked up automatically |
| NVIDIA GPU + driver | training, DNSMOS scoring | CUDA libraries come from pip wheels; only the driver is needed on the system. Developed on an RTX 4070 SUPER (12 GB); `batch_size = 32` with 3 s segments uses about 3 GB |
| C compiler (`sudo apt install build-essential`) | PESQ (the `pesq` package builds from source), the C++ core | without it PESQ is reported as NaN, everything else works |
| g++ (C++20) and make | `cpp/` | no other C++ dependency |
| Windows + Visual Studio (MSVC, CMake, Ninja) | the VST3 plugin only | see section 8 |
| Disk | data | the raw downloads are large (WHAM! alone is a 68 GB zip, Expresso 38 GB, DNS Challenge noise about 39 GB); plan for several hundred GB including the staged FLAC files |

### 1.2 Create the environments

```bash
uv sync                                  # main environment: data pipeline + evaluation (onnxruntime-gpu, pystoi, pesq)
uv sync --project train                  # training environment: Keras 3 + TensorFlow (CUDA 12 wheels)
```

The two environments are deliberately separate: TensorFlow brings CUDA 12 libraries, `onnxruntime-gpu` brings CUDA 13 ones, and mixing them in one environment can make them overwrite each other.

You never activate the train environment by hand. The commands `train`, `infer`, `export`, `check ...` and `tensorboard` re-launch themselves inside it (see `src/transnoise/train_env.py`), after putting the pip-installed CUDA libraries on `LD_LIBRARY_PATH`; otherwise TensorFlow silently falls back to the CPU. Do not run the training code with a plain `uv run python`.

### 1.3 Evaluation assets and baseline models (needed for section 6)

```bash
bash scripts/fetch_eval_assets.sh        # DNSMOS model, DeepFilterNet3 binary + ONNX model (skips files that already exist)
uv sync --project models/rnnoise         # RNNoise baseline environment
uv sync --project models/mossformer2     # MossFormer2_SE_48K baseline environment (clearvoice pins old numpy/librosa, hence its own environment)
```

The baselines are optional; skip the ones you do not want to compare against. The DNSMOS model (`data/eval/assets/dnsmos/sig_bak_ovr.onnx`) is required for scoring.

### 1.4 Sanity check

```bash
uv run transnoise --help                 # lists data, eval, train, infer, export, check, tensorboard
uv run transnoise check model            # builds the model with random weights and verifies shapes, perfect STFT reconstruction and causality
```

`check model` ends with `全部通过` ("all passed") when everything is fine.

---

## 2. Data pipeline

The pipeline turns raw downloads into clean 48 kHz mono FLAC files plus an index. Mixing speech with noise happens later, online during training (section 5), so the pipeline only produces clean material.

### 2.1 Data sources

```bash
uv run transnoise data list
```

| Name | Role | Language | Notes |
|---|---|---|---|
| `libritts_r` | speech | en | 24 kHz, upstream train/dev/test split |
| `vctk` | speech | en | native 48 kHz; an 11.7 GB outer archive is downloaded and the inner zip extracted |
| `ears` | speech | en | 48 kHz expressive speech (CC BY-NC) |
| `expresso` | speech | en | expressive acted speech, a single 38 GB tar; channel 0 only (CC BY-NC) |
| `aishell1` | speech | zh | 16 kHz, upstream split |
| `aishell3` | speech | zh | 44.1 kHz |
| `rnnoise_noise` | noise | - | official RNNoise noise (headerless int16 48 kHz `.sw` files) |
| `demand` | noise | - | only channel 1 of the 16-channel array is used |
| `dns5_noise` | noise | - | DNS Challenge 5 noise: AudioSet (CC BY) + Freesound (CC0 only), about 39 GB of bz2 archives |
| `wham_noise` | noise | - | WHAM! 48 kHz binaural noise, left channel only (CC BY-NC) |
| `rirs_simulated`, `rirs_real` | RIR | - | RIRs & Noises; both come from the same archive, which is downloaded once through `rirs_simulated` |

Check each license before downloading or redistributing anything; several sources are non-commercial.

### 2.2 Download, stage, index

```bash
# 1. download (segmented multi-connection, resumable; re-run the same command after an interruption)
uv run transnoise data download libritts_r vctk ears expresso aishell1 aishell3 \
    rnnoise_noise demand dns5_noise wham_noise rirs_simulated

# 2. stage: decode, clean, resample to 48 kHz mono FLAC, write manifests (also resumable)
uv run transnoise data stage libritts_r vctk ears expresso aishell1 aishell3 \
    rnnoise_noise demand dns5_noise wham_noise rirs_simulated rirs_real

# 3. inspect what was kept and what was dropped
uv run transnoise data status

# 4. merge the manifests into data/index/{train,val,test}.jsonl
uv run transnoise data index
```

You can run any subset first (for example a single speech source plus `demand` and `rirs_simulated`) to try the pipeline quickly; training requires at least one speech, one noise and one RIR source in the index.

Options and behavior:

- `data download <sources...> [--conns N]`: connections per file (default 32, from `configs/data_pipeline.toml`).
- `data stage <sources...> [--workers N] [--redo]`: `--redo` deletes that source's manifest, completion marks and `data/staged/<source>/` and processes it again (`raw/` is untouched). Do not `--redo` a source that is already in use unless you really want it re-split: the split is recomputed from the current ratios.
- Cleaning (all thresholds in `configs/data_pipeline.toml`): too short, silent, clipped speech, and noise or speech with an effective bandwidth below the source's `min_bandwidth_hz` are dropped. Every kept speech file gets a measured bandwidth (`bandwidth_hz`) and a tier A/B/C; training uses it to mask the loss above each clip's cutoff.
- Long recordings (WHAM!, Expresso) are cut into 60 s segments; WHAM! keeps only the left channel because a stereo downmix causes comb filtering at high frequencies.
- Output layout:

```
data/raw/        original downloads, never modified
data/staged/     48 kHz mono FLAC (RIRs are float32 WAV)
data/manifests/  one jsonl per source (includes dropped rows) + units.json (finished archives)
data/index/      train/val/test.jsonl, kept rows only; training reads train/val, only evaluation reads test
```

### 2.3 Split rules

- Split labels are written into the manifest at `stage` time; nothing is moved between directories.
- Speech is split by speaker, RIRs by room, noise by source-file hash; one group never crosses train/val/test. For RIRs the `speaker` field holds the room id; for WHAM! it holds the recording location, and the upstream split is used.
- LibriTTS-R and AISHELL-1 keep their upstream splits regardless of the `[split]` ratios in the config.
- Never train on the test split.

---

## 3. Evaluation test sets

Two fixed test sets are built from the `split=test` part of the index, with fixed seeds:

| Name | Content |
|---|---|
| `noise` | noisy speech, reference = clean speech |
| `reverb` | speech convolved with an RIR first; reference = reverberant clean speech (only denoising is evaluated, not dereverberation) |

Each has 600 clips (300 English, 300 Chinese; SNR -5, 0, 5, 10, 15, 20 dB with equal counts; 3-10 s each).

```bash
uv run transnoise eval build-testset noise
uv run transnoise eval build-testset reverb
```

- Build them once; a test set that already exists is refused. After the index changes, rebuild with `--force` (this also deletes the old scores, enhanced wavs and reports of that test set, because they no longer match).
- Files are written to `data/eval/testsets/<name>/{noisy,clean,meta.jsonl,spec.toml}`. `spec.toml` is a copy of the generation parameters from `configs/eval.toml`.

---

## 4. Self-checks before training (optional but recommended)

```bash
uv run transnoise check data        # reads the real index: shapes, SNR accuracy, sample-kind ratios, reverb ratio, determinism, resume position, throughput, noise source overview
uv run transnoise check training    # synthetic data: bandwidth mask, overfit one batch, resumed training is bit-identical, NaN protection
uv run transnoise check batch       # measures peak GPU memory and speed for segment length x batch size, to choose batch_size
```

Each prints `[通过]` (pass) or `[失败]` (fail) per item and ends with `全部通过` or `有失败项` (there are failures); the exit code is non-zero on failure.

---

## 5. Training

### 5.1 What gets trained

- **Model** (`configs/model.toml`, the only place where model hyperparameters are defined): STFT 960/480 -> 32 ERB features + low-frequency complex features -> sliding-window causal Transformer -> ERB gains + low-frequency deep filter -> inverse STFT. Low band output (bins below `nb_df`) = deep filter (applied to the noisy spectrum) x ERB gain, which acts as a 0-1 gate; the high band is multiplied by the gain only.
- **Lookahead**: `lookahead` frames (10 ms each) are used only in the first attention layer and the deep filter; later layers are strictly causal, so the total lookahead does not grow with depth. `df_order` must be greater than `lookahead`. Latency is (lookahead + 1) x 10 ms overall and at most (lookahead + 2) x 10 ms; `lookahead = 0` is fully causal.
- **Loss**: multi-resolution spectral loss (FFT 256/512/1024/2048, compression exponent 0.3, a magnitude term and a complex term). Every sample carries a cutoff frequency (the measured bandwidth of its clean speech); bins above it do not count.
- **Optimizer**: AdamW, linear warmup then cosine decay, global gradient-norm clipping, EMA weights (used for evaluation and export). No weight decay on biases, LayerNorm and the relative position bias.
- **Data** (`training/dataset.py`, `[data]` in `configs/train.toml`): everything is mixed online, every step draws speech x noise x (RIR) x SNR x level. Sample kinds: `mixed` (speech + noise), `noise_only` (target all zeros), `clean_only` (target = input) and `silence` (all zeros), with fractions `p_noise_only`, `p_clean`, `p_silence` (the rest is `mixed`). Speech is drawn uniformly over files; all noise sources form one pool sampled by file duration; RIRs pick a source first, then a file. Batch number `step` is determined by `(seed, step)` alone, so resuming only needs the step.

All training hyperparameters live in `configs/train.toml`.

### 5.2 Run it

```bash
# choose a run name; the checkpoint directory is data/checkpoints/<name>/
nohup uv run transnoise train --name v1 > scratch/train_v1.log 2>&1 &
tail -f scratch/train_v1.log

uv run transnoise tensorboard            # curves at http://localhost:6006
```

(`scratch/` is git-ignored and is the place for logs and one-off scripts.)

Options:

| Command | Meaning |
|---|---|
| `train` | train; resumes automatically from the latest checkpoint in the run directory |
| `train --name exp1` | use another run directory (default name is `run.name` in `configs/train.toml`) |
| `train --set batch_size=16 --set peak_lr=3e-4` | override hyperparameters for this run only (field names of `TrainConfig`) |
| `train --max-steps 300` | smoke test: stop after 300 steps in this invocation |
| `train --rerun` | delete everything in the run directory (checkpoints, best, logs, curves) and start from scratch; it lists what will be deleted and asks for confirmation, add `--yes` in scripts or under `nohup` |

Things to know:

- **Do not resume an old run after changing the model or the data setup.** A mismatched architecture is refused with an error, but a changed data distribution is not detected; use a new `--name` (or `--rerun`).
- Ctrl+C and `kill` (SIGTERM) save a checkpoint before exiting; re-run the same command to continue.
- The run directory contains `config_model.toml` and `config_train.toml`, snapshots of the settings used; inference, export and deployment read the model structure from the snapshot, so later edits to `configs/model.toml` do not break old checkpoints.
- Logs go to the terminal, `logs.csv` (training) and `val.csv` (validation) in the run directory, and TensorBoard (`tb/`, written only if `tensorboard` is installed). `data_wait` is the fraction of time spent waiting for data (near 0 means the GPU is the bottleneck); `grad_norm` is the global gradient norm before clipping.
- Every `eval_every` steps the raw and the EMA weights are evaluated on a fixed validation set (16 fixed batches); when the EMA loss reaches a new low, `best/` is saved. `best/ema.npz` is the weights to use. Validation losses are only comparable between runs that use the same data setup.
- Steps with a NaN/Inf loss or gradient are skipped entirely; after `nan_patience` such steps in a row the run rolls back to the previous checkpoint (at most `max_rollbacks` times).
- Checkpoints: `step_XXXXXXXX/` (model + optimizer + loop state), written to a temporary directory and renamed, the last 3 are kept, `best/` is stored separately.
- Training time observed on an RTX 4070 SUPER for the default 100000 steps: roughly 4.5-5 hours.

---

## 6. Inference and scoring

The evaluation interface is fixed: **input = a test set + a directory of enhanced wavs; output = one row of scores per clip.** It does not know which model produced the wavs. Requirements for the directory: every `<id>.wav` of the test set exists, 48 kHz mono. Delay needs no handling: the evaluation measures it by cross-correlation (+-500 ms) and aligns automatically; the measured delay is reported in the table.

By convention the enhanced wavs go to `data/eval/runs/<testset>/<name>/wav`, and the result name is taken from the directory name (the parent's name when the directory is called `wav`).

### 6.1 Your model

```bash
for ts in noise reverb; do
  uv run transnoise infer --ckpt data/checkpoints/v1/best \
      --in data/eval/testsets/$ts/noisy --out data/eval/runs/$ts/ours/wav
done
```

`--ckpt` accepts a run directory (uses `latest`), `<run>/best`, or `<run>/step_XXXXXXXX`; EMA weights are used unless you pass `--raw`. Inference is offline over whole clips, so the output is time-aligned with the input.

### 6.2 References and baselines

```bash
for ts in noise reverb; do
  T=$PWD/data/eval/testsets/$ts; R=$PWD/data/eval/runs/$ts
  # RNNoise
  models/rnnoise/.venv/bin/python models/rnnoise/run.py --in $T/noisy --out $R/rnnoise/wav
  # DeepFilterNet3 (precompiled binary; fetched by scripts/fetch_eval_assets.sh)
  python3 models/dfn3/run.py --in $T/noisy --out $R/dfn3/wav
  # MossFormer2_SE_48K (must run with its own directory as cwd; use absolute paths)
  (cd models/mossformer2 && .venv/bin/python run.py --in $T/noisy --out $R/mossformer2/wav)
done
```

These tools are one-off scripts and not part of the evaluation. The MossFormer2 weights are downloaded by `clearvoice` on first use (and kept under `models/mossformer2/checkpoints/`, git-ignored).

### 6.3 Score and report

```bash
for ts in noise reverb; do
  for name in ours rnnoise dfn3 mossformer2; do
    uv run transnoise eval score $ts data/eval/runs/$ts/$name/wav --name $name
  done
  # the unprocessed input is the baseline for dSI-SDR; the clean reference should score perfectly (self-check)
  uv run transnoise eval score $ts data/eval/testsets/$ts/noisy --name noisy
  uv run transnoise eval score $ts data/eval/testsets/$ts/clean --name clean
  uv run transnoise eval report $ts
done
```

- `eval score <testset> <dir> [--name N] [--workers N]` takes about 2.5 minutes per directory. Scores are written to `data/eval/runs/<testset>/<name>/results.jsonl` and always recomputed from scratch.
- `eval report <testset> [names...] [--by lang,snr_db,speech_source,noise_source] [--metric si_sdr_imp]` writes `data/eval/reports/<testset>.md` and `.csv`: an overall table (mean +- 95% confidence interval) and slices by the given fields. The result named `noisy` is the baseline for dSI-SDR; `clean` is excluded from SI-SDR comparison.
- Metrics: SI-SDR (48 kHz), PESQ-wb, STOI, DNSMOS P.835 (SIG/BAK/OVRL, at 16 kHz). Without the `pesq` package PESQ is NaN and the rest is unaffected.
- For the `reverb` set, SI-SDR/PESQ/STOI penalize models that also dereverberate; do not rank by them, look at DNSMOS.

---

## 7. Deployment to C++

```bash
RUN=v1

# 1. export the checkpoint to named tensors (EMA weights) -> data/deploy/$RUN/weights.npz
uv run transnoise export --ckpt data/checkpoints/$RUN/best

# 2. pack for C++ -> cpp/gen/weights.bin + cpp/gen/config.h
uv run --no-sync python -m transnoise.deploy.pack data/deploy/$RUN/weights.npz --out-dir cpp/gen

# 3. generate the cross-check fixture -> cpp/gen/fixture.bin (needs the `noise` test set; see --wav to use another file)
uv run --no-sync python -m transnoise.deploy.fixture data/deploy/$RUN/weights.npz --out cpp/gen/fixture.bin

# 4. numpy streaming reference vs the Keras offline forward
uv run transnoise check deploy --ckpt data/checkpoints/$RUN/best

# 5. C++ vs the numpy reference, per layer and on the output waveform; also checks BlockAdapter with many block sizes
cd cpp && make test && make tn_wav && cd ..
```

What to expect:

- `check deploy` passes when the numpy streaming output matches Keras with a relative RMS error below 1e-4 (in practice about 5e-7), a deliberately broken control gives a much larger error, and very short inputs also agree.
- `make test` passes when every per-frame intermediate is within its threshold, the output waveform error is below 1e-5, and every block size gives bit-identical output with the documented latency. It also prints the per-frame time (a few hundred microseconds per 10 ms frame in practice).
- `fixture.bin` is derived data and is git-ignored; `weights.bin` and `config.h` are tracked because the plugin build embeds them.

Notes:

- `pack` folds every LayerNorm's gamma and beta into the following linear layer, and writes `config.h`: all shapes in C++ are compile-time constants, and the header of `weights.bin` carries the same constants and is checked item by item on load. **Changing the architecture means retraining, re-packing and recompiling.**
- Latency reported by `BlockAdapter`: for a fixed host block size B it is (LA+1)*HOP + HOP - gcd(B, HOP) samples (960 samples = 20 ms when B is a multiple of 480); for variable block sizes it is (LA+2)*HOP - 1 samples (about 30 ms).
- `cpp/include/tn/`: `model.h` (`Weights`, `Denoiser`: frame-by-frame inference with a KV cache; the real-time path allocates nothing, takes no locks and throws nothing), `fft.h`, `stream_adapter.h` (`BlockAdapter`: any host block size <-> the model's HOP frames, with a fixed latency), `wav.h`.
- `src/transnoise/deploy/ref.py` is the numpy streaming reference (numpy only); every C++ layer is checked against it.

### Scoring the C++ output (optional)

```bash
cpp/build/tn_wav --weights cpp/gen/weights.bin \
    --in data/eval/testsets/noise/noisy --out data/eval/runs/noise/cpp/wav [--block 480] [--limit N]
uv run transnoise eval score noise data/eval/runs/noise/cpp/wav --name cpp
```

`--block` simulates the host block size (`0` = variable block sizes, worst-case latency). The C++ output should score the same as the Python one.

---

## 8. VST3 plugin (Windows)

Built with JUCE (AGPLv3) + CMake + MSVC. The build tree and the JUCE sources live in `plugin/build/` (git-ignored); JUCE is downloaded by CMake as a source archive, so the first build needs an internet connection.

Prerequisites on Windows: Visual Studio with the C++ toolchain (the script expects `C:\Program Files\Microsoft Visual Studio\18\Community`; edit the `VS` variable at the top of `plugin/build_win.bat` if yours is elsewhere). It uses the CMake and Ninja bundled with Visual Studio.

### 8.1 Build

First make sure `cpp/gen/weights.bin` and `cpp/gen/config.h` exist for the model you want (section 7, steps 1-2). Then run in a Windows `cmd` (the project lives on the WSL filesystem; `pushd` maps the `\\wsl.localhost\...` path to a temporary drive letter, which cmd, MSVC and Ninja need):

```
cmd /c "pushd \\wsl.localhost\<distro>\<path-to-repo>\plugin && .\build_win.bat"
```

(Replace `<distro>` with your WSL distribution name and `<path-to-repo>` with the repository path inside it, using backslashes. If the repository is on a Windows drive, just `cd plugin` and run `build_win.bat`.) The first build takes about 2 minutes, later ones are incremental. The result:

```
plugin\build\Transnoise_artefacts\Release\VST3\Transnoise.vst3
```

A standalone application is built next to it for quick testing.

### 8.2 Install and use

1. Copy `Transnoise.vst3` to `C:\Program Files\Common Files\VST3\` (the standard system VST3 folder) and rescan plugins in your host.
2. Insert it on a mono or stereo track or bus, **running at 48 kHz**; at other sample rates the plugin passes audio through untouched (and reports no latency).
3. Parameters: **Mix** (dry/wet, 0-100 %) and **Bypass**. The dry signal is delayed by exactly the plugin latency, so neither Mix nor Bypass introduces a time or phase misalignment.
4. Latency: the plugin reports the worst-case latency for variable block sizes (about 30 ms) to the host, which lets the host compensate it.
5. Each channel has its own model instance (the weights are shared and compiled into the plugin).

If the weights fail to load, the plugin passes audio through instead of crashing.

---

## 9. Configuration reference

| File | Controls |
|---|---|
| `configs/model.toml` | model structure: STFT sizes, ERB bands, `nb_df`, Transformer width/depth/heads, `window`, `lookahead`, deep-filter `order` |
| `configs/train.toml` | run name and seed, segment length, batch size, loss, optimizer, checkpointing, online-mixing data settings (SNR range, speech level, sample-kind fractions), training loop |
| `configs/data_pipeline.toml` | download connections/retries, bandwidth tiers, cleaning thresholds, split ratios, staging workers, `.sw` segmenting |
| `configs/eval.toml` | test sets (counts, SNRs, durations, seeds), metric settings, DNSMOS model path |
| `src/transnoise/data_pipeline/registry.py` | the list of data sources (URLs, license, per-source options) |

Hyperparameters are defined only in these files; the code has no duplicated defaults.

## 10. Repository layout

```
configs/                 all hyperparameters (see above)
src/transnoise/
  cli.py                 the `transnoise` command
  paths.py, mixing.py    data locations; noisy-speech synthesis shared by training and evaluation
  data_pipeline/         download, stage, index
  eval/                  test sets, metrics, scoring, reports
  model/                 STFT front end, attention, heads, the Denoiser (Keras)
  training/              dataset, losses, optimizer, trainer, training loop, inference
  deploy/                export, pack, fixture, numpy streaming reference
train/                   train-environment project and self-check scripts (check_*.py, measure_batch.py)
cpp/                     C++ core (include/tn), tests, command-line tool, generated gen/
plugin/                  JUCE VST3 plugin
models/                  baseline models used for comparison (one-off inference tools)
scripts/                 helper scripts (fetch_eval_assets.sh)
scratch/                 git-ignored: temporary scripts, logs, intermediate files
data/                    git-ignored: raw/, staged/, manifests/, index/, checkpoints/<run>/, deploy/<run>/, eval/
```

## 11. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Training is very slow, `nvidia-smi` shows no use | TensorFlow fell back to the CPU: run through `uv run transnoise train ...` (not `python` directly) so `LD_LIBRARY_PATH` is set; check the `train` environment is synced |
| PESQ column is `NaN` | the `pesq` package is not installed (needs a C compiler); `sudo apt install build-essential`, then `uv sync` |
| DNSMOS warns it is running on the CPU | the CUDA libraries of the main environment are missing; run `uv sync` and check the NVIDIA driver |
| `eval score` says files are missing | the directory must contain `<id>.wav` for every clip of the test set |
| Old scores no longer match | the test set was rebuilt (`--force`) or the index changed; re-run inference and scoring |
| Refusing to load a checkpoint ("structure mismatch") | `configs/model.toml` was changed after training; use the run's `config_model.toml` snapshot (inference and export do this automatically) or retrain |
| `weights.bin` refused by C++ | `weights.bin` and `config.h` come from different models, or `config.h` was not recompiled; re-run `pack` and rebuild |
| `stage` was interrupted | just run the same command again; it resumes. Use `--redo` only to start a source from scratch |
| Download interrupted | run the same `download` command again; chunks already fetched are kept |
| Plugin build cannot find `weights.bin` | run `export` and `pack` first (section 7) |
| Plugin does nothing | the host is not running at 48 kHz, or the weights failed to load; the plugin then passes audio through |

## 12. License

| Part | License |
|---|---|
| Everything except `plugin/` (C++ core in `cpp/`, training, data pipeline, evaluation, scripts) | MIT, see `LICENSE` |
| `plugin/` (the VST3 / standalone plugin, built with JUCE) | AGPLv3, see `plugin/LICENSE` |

The plugin links the MIT-licensed core, which is allowed; the combined plugin binary is distributed under the AGPLv3, and so are its sources. Third-party data sources have their own licenses (listed in `src/transnoise/data_pipeline/registry.py`); trained weights inherit the restrictions of the data they were trained on.
