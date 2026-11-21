# noise-cancel

语音降噪：数据管线（已完成）、评测、模型（待做）。在 WSL 里用 uv 运行。

## 数据管线

```
uv run noise-cancel data list                      # 列出数据源
uv run noise-cancel data download phase1 phase2    # 下载到 data/raw/（分段多连接，可断点续传）
uv run noise-cancel data stage phase1 phase2       # 重采样 48kHz 单声道 FLAC、清洗，写 data/manifests/
uv run noise-cancel data status                    # 按来源/语言/带宽/切分统计
uv run noise-cancel data index                     # 汇总成 data/index/{train,val,test}.jsonl
```

- 参数在 `configs/data_pipeline.toml`；数据源清单在 `src/noise_cancel/data_pipeline/registry.py`。
- `download --mirror elda`：OpenSLR 主站对单连接限速，镜像更快。
- `stage` 随时可中断，重跑会续；`stage <源> --redo` 清空该源的处理结果后重做（`raw/` 不动）。

## 目录

```
data/raw/        原始下载，不改动
data/staged/     48kHz 单声道 FLAC（RIR 为 float32 WAV）
data/manifests/  每个数据源一个 jsonl（含被丢弃的行）+ units.json（已完成的压缩包）
data/index/      train/val/test.jsonl，只含保留的行；训练读 train/val，评测才读 test
```

## 切分规则

- 切分标签在 `stage` 时写进 manifest，不物理分目录；训练时在内存里按 `role`/`lang`/`split` 过滤并加权采样。
- 语音按说话人、RIR 按房间、噪声按源文件哈希，同一组不会跨 train/val/test。`speaker` 字段对 RIR 存的是房间 id。
- LibriTTS-R 和 AISHELL-1 沿用上游切分，不受配置里 `[split]` 比例影响。
- 已经在用的数据源不要随意 `--redo`，否则会按当前比例重新划分。
