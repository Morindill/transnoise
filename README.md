# transnoise

流式语音降噪：数据管线、评测、模型训练（Keras 3 + TensorFlow）、C++ 推理核心、VST3 插件。

| 部分 | 位置 | 运行环境 |
|---|---|---|
| 数据管线、评测 | `src/transnoise/{data_pipeline,eval}` | WSL，主 uv 环境 |
| 模型、训练 | `src/transnoise/{model,training}`、`train/` | WSL，`train/` 独立 uv 环境 |
| 部署导出、numpy 参考实现 | `src/transnoise/deploy/` | 导出在 train 环境，其余在主环境 |
| C++ 推理核心 | `cpp/` | WSL，g++ + make |
| VST3 插件 | `plugin/` | Windows，MSVC + CMake |

## 数据管线

```
uv run transnoise data list                      # 列出数据源
uv run transnoise data download <源名...>        # 下载到 data/raw/（分段多连接，可断点续传）
uv run transnoise data stage <源名...>           # 重采样 48kHz 单声道 FLAC、清洗，写 data/manifests/
uv run transnoise data status                    # 按来源/语言/带宽/切分统计
uv run transnoise data index                     # 汇总成 data/index/{train,val,test}.jsonl
```

- 参数在 `configs/data_pipeline.toml`；数据源清单在 `src/transnoise/data_pipeline/registry.py`。
- 数据源：语音 LibriTTS-R、VCTK、EARS、Expresso（英）、AISHELL-1、AISHELL-3（中）；噪声 RNNoise 官方噪声、DEMAND、WHAM!、DNS Challenge 5 噪声；RIR 来自 RIRs & Noises（模拟 + 真实）。
- `download --mirror elda`：OpenSLR 主站对单连接限速，镜像更快。
- `stage` 随时可中断，重跑会续；`stage <源> --redo` 清空该源的处理结果后重做（`raw/` 不动）。
- WHAM!：68 GB 单个 zip；只取左声道（双耳录音下混会在高频造成梳状滤波）、切 60 秒一段、带宽 < 10 kHz 的丢弃；切分用上游的按地点划分，`speaker` 字段存地点 id。`min_bandwidth_hz` 门槛只对设了它的来源生效。

## 目录

```
data/raw/        原始下载，不改动
data/staged/     48kHz 单声道 FLAC（RIR 为 float32 WAV）
data/manifests/  每个数据源一个 jsonl（含被丢弃的行）+ units.json（已完成的压缩包）
data/index/      train/val/test.jsonl，只含保留的行；训练读 train/val，评测才读 test
data/checkpoints/<run>/   训练存档
data/deploy/<run>/        导出的部署权重（weights.npz）
data/eval/       评测资源、测试集、结果、报告
```

## 切分规则

- 切分标签在 `stage` 时写进 manifest，不物理分目录；训练时在内存里按 `role`/`lang`/`split` 过滤并加权采样。
- 语音按说话人、RIR 按房间、噪声按源文件哈希，同一组不会跨 train/val/test。`speaker` 字段对 RIR 存的是房间 id。
- LibriTTS-R 和 AISHELL-1 沿用上游切分，不受配置里 `[split]` 比例影响。
- 已经在用的数据源不要随意 `--redo`，否则会按当前比例重新划分。

## 评测

评测的输入输出是固定的：**输入 = 测试集 + 一个增强后的 wav 目录，输出 = 每条一行的分数**。评测不知道 wav 是哪个模型产生的；
各基线模型的推理工具在 `models/<名>/run.py`，是一次性工具，不属于评测。

```
uv run transnoise eval build-testset noise            # 合成测试集（固定种子；同名已存在会拒绝重建，--force 重建并删除该测试集的评分/增强 wav/报告）
uv run transnoise eval score noise <增强wav目录>       # 打分（约 2.5 分钟）；--name 指定结果名，默认取目录名（目录叫 wav 时取上一级）
uv run transnoise eval report noise [名字...]          # 对比；--by 切片字段，--metric 切片表的指标
```

- 测试集在 `data/eval/testsets/<名>/{noisy,clean,meta.jsonl,spec.toml}`，来自 index 的 `split=test`，每个 600 条（中英各 300，SNR -5…20dB 各 100）。index 变了测试集要 `--force` 重建，否则和 index 对不上。
- wav 目录要求：测试集每条的 `<id>.wav` 都在，48kHz 单声道。延迟不用处理：评测用互相关测出延迟（±500ms）并对齐，记在"延迟 ms"一列。
- 结果在 `data/eval/runs/<测试集>/<名字>/results.jsonl`，报告在 `data/eval/reports/`。
- 约定：名为 `noisy` 的结果（给 `testsets/<名>/noisy` 目录打分）是 ΔSI-SDR 的基线；给 `clean` 目录打分应得满分，用来自检。
- 指标：SI-SDR（48k）、PESQ-wb、STOI、DNSMOS P.835（SIG/BAK/OVRL，16k）。PESQ 缺失（没装 pesq）时记 NaN，其余照常。
- `reverb` 测试集的参考是"带混响的干净语音"，会惩罚有去混响效果的模型，SI-SDR/PESQ/STOI 别直接当排名，看 DNSMOS。
- 不要用测试集训练。

## 模型（Keras 3 + TensorFlow）

训练环境独立在 `train/`（TF 带 CUDA 12 库，不能和评测环境的 onnxruntime-gpu 混装）。
```
uv sync --project train                       # 安装
uv run transnoise check model            # 模型自检（随机权重，不训练）
```
- 这些命令会在 train 环境里重启自己，并把 pip 装的 CUDA 库加进 `LD_LIBRARY_PATH`（见 `src/transnoise/train_env.py`），否则 TF 会悄悄退回 CPU；不要直接用 `uv run python` 跑训练代码。
- 超参在 `configs/model.toml`（所有超参只在这里定义）；代码在 `src/transnoise/model/`（只在 train 环境里导入，主项目不依赖 TF）。

结构：STFT(960/480) → ERB 特征(32) + 低频复数特征(nb_df) → 滑动窗口因果 Transformer → ERB 增益 + 低频深度滤波（低频输出 = 深度滤波 × ERB 增益，高频只有增益）→ 逆 STFT。
- 前视 `lookahead` 帧只放在第一层注意力和深度滤波里，后面的层严格因果，总前视不随层数累加。
- 延迟：整体 (lookahead+1)×10 ms，最坏 (lookahead+2)×10 ms；`lookahead=0` 是完全因果。
- `df_order=0` 关闭深度滤波；`window` 是每帧向过去看多少帧。

### 训练核心（`src/transnoise/training/`）
```
uv run transnoise check training                      # 自检：带宽掩码、过拟合一个 batch、断点续训逐位一致
uv run transnoise check batch                         # 实测 片段长度 × batch 的显存峰值和速度，用来定 batch_size
```
- 损失：多分辨率谱损失（FFT 256/512/1024/2048，压缩指数 0.3，幅度项 + 复数项），**每条样本带截止频率**，高于它的频点不计入（对应数据管线里的带宽分级）。
- 优化器：AdamW + 线性预热 → 余弦退火，梯度全局范数裁剪；bias / LayerNorm / 相对位置偏置不做权重衰减。
- 断点：`data/checkpoints/<run>/step_XXXXXXXX/`（模型 + 优化器状态 + 训练循环状态），先写临时目录再改名，滚动保留 3 个，`best/` 另存；恢复时校验形状，结构对不上直接报错。

### 训练数据（`training/dataset.py`，在线混合）
```
uv run transnoise check data              # 自检：SNR/混响比例/确定性/续训位置/吞吐/噪声文件数概览（读真实 index）
```
- 每步现场抽 语音 × 噪声 × (RIR) × SNR × 电平，不预先混好；`data_pipeline` 只产出干净素材。参数在 `configs/train.toml` 的 `[data]`。
- 语音按 index 里的条数均匀抽（不做中英加权）；噪声把所有来源并成一个集合，按文件时长加权抽（全部噪声音频的时间轴上均匀）；RIR 先均匀选来源再选文件。
- 混合运算在 `src/transnoise/mixing.py`，评测测试集也用它。
- 第 `step` 个 batch 只由 `(seed, step)` 决定，续训只需要 step，不用存采样器状态。
- 带混响样本的目标是"带混响的干净语音"（只去噪）。每条样本的 `cutoff_hz` = 该语音的 `bandwidth_hz`，给损失做带宽掩码。

### 训练循环（`training/run.py`）
```
uv run transnoise train                       # 开始训练；有检查点自动续训
uv run transnoise train --name exp1 --set batch_size=16 --set peak_lr=3e-4   # 换名字/临时覆盖超参
uv run transnoise train --max-steps 300       # 冒烟测试：这次只训 300 步
uv run transnoise tensorboard                     # 看曲线，浏览器开 http://localhost:6006
uv run transnoise infer --ckpt data/checkpoints/<run>/best --in 带噪目录 --out 输出目录   # 用检查点增强一个目录的 wav（EMA 权重，离线整段）
```
- 长跑放进 tmux 或 `nohup ... &`；Ctrl+C / `kill`（SIGTERM）会先存档再退出，重新运行同一条命令即可接上。
- 日志：终端 + `data/checkpoints/<name>/logs.csv`（训练）和 `val.csv`（验证）+ TensorBoard（`tb/`，装了 tensorboard 才写）。
  `data_wait` 是等数据的时间占比，接近 0 说明瓶颈在 GPU；`grad_norm` 是裁剪前的全局梯度范数。
- 每 `eval_every` 步在固定验证集上评估原始/EMA 两套权重，EMA 更优就存 `best/`（里面的 `ema.npz` 才是要导出的权重）。
- NaN/Inf 的步整步跳过；连续 `nan_patience` 步就回滚到上一个检查点。
- 运行目录里有 `config_*.toml`，是这次运行的配置快照。
- `--rerun`：删除这个运行目录里的全部存档（检查点、best、日志、曲线）后从头训练；会先列出要删的内容并要求确认，`nohup`/脚本里加 `--yes`。

## 部署（`src/transnoise/deploy/`、`cpp/`）

```
uv run transnoise export --ckpt data/checkpoints/<run>/best              # → data/deploy/<run>/weights.npz（按结构命名的张量）
uv run --no-sync python -m transnoise.deploy.pack data/deploy/<run>/weights.npz --out-dir cpp/gen   # → weights.bin + config.h
uv run --no-sync python -m transnoise.deploy.fixture data/deploy/<run>/weights.npz --out cpp/gen/fixture.bin   # C++ 对拍用的夹具
uv run transnoise check deploy                         # numpy 参考实现 vs Keras 离线前向（必须关 TF32，脚本里已关）
cd cpp && make test                                          # C++ vs numpy 参考实现：逐层中间量、输出波形、BlockAdapter 分块
cd cpp && make tn_wav                                        # 命令行：增强一个目录的 wav，结果可直接给 eval score 打分
```

- `deploy/ref.py` 是带 KV 缓存的 numpy 流式参考实现，只依赖 numpy；C++ 的每一层都和它对拍。
- `pack` 把每个 LayerNorm 的 γ、β 折进后面的线性层，并生成 `config.h`：C++ 里所有形状都是编译期常量，`weights.bin` 的头里带同一组常量，加载时逐项校验，对不上就拒绝加载。换模型结构要重新 pack 并重新编译。
- `cpp/include/tn/`：`model.h`（`Weights`、`Denoiser`，逐帧推理，实时路径不分配内存）、`fft.h`、`stream_adapter.h`（`BlockAdapter`：任意宿主块大小 ↔ 模型的 HOP 帧，给出固定延迟 `latency()`）、`wav.h`。
- 延迟：固定块 B 时 (LA+1)·HOP + HOP − gcd(B, HOP) 个样本；块大小不固定时 (LA+2)·HOP − 1。
- 构建只需要 g++（C++20）和 make，没有第三方依赖。

## VST3 插件（`plugin/`）

JUCE（AGPLv3）+ CMake，在 Windows 上用 MSVC 构建，构建树和 JUCE 源码都在工程目录里的 `plugin/build/`（已被 git 忽略）。在 Windows 的 cmd 里运行（`pushd` 会把 `\\wsl.localhost` 路径临时映射成盘符，cmd、MSVC 和 Ninja 需要这个）：
```
cmd /c "pushd \\wsl.localhost\Ubuntu\home\morindill\noise_cancel\plugin && .\build_win.bat"
```
产物在 `plugin/build/Transnoise_artefacts/Release/VST3/Transnoise.vst3`（首次约 2 分钟，之后增量）。JUCE 用 CMake 从 GitHub 下载源码压缩包，需要联网。

- 单声道/立体声，每个声道一个模型实例，共用一份权重（编进插件）；参数 Mix（干湿）、Bypass，干声按插件延迟对齐。
- 只支持 48 kHz：其它采样率下直通。报告给宿主的延迟是块大小不固定时的最坏情况。
- 构建需要 `cpp/gen/{config.h,weights.bin}`（见上面的部署步骤）。
