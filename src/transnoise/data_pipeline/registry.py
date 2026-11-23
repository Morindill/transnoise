"""数据源注册表。许可证请下载前自行复核。"""

from dataclasses import dataclass
from pathlib import Path

from ..paths import RAW


@dataclass(frozen=True)
class Source:
    name: str
    role: str  # speech | noise | rir
    lang: str  # en | zh | -
    license: str
    urls: tuple[str, ...] = ()
    raw_name: str = ""  # 多个 Source 共用同一份原始包时指定（如 rirs_noises）
    kind: str = "archive"  # archive: tar/zip；raw: 无头 int16 48k（.sw）
    include: str = ""  # 对包内路径的正则，空 = 所有音频
    speaker: str = ""  # 含命名组 spk 的正则；空 = 无说话人信息
    split: str = ""  # 含命名组 split(train|dev|test) 的上游切分；空 = 哈希
    unwrap: str = ""  # 下载到的是外层 zip 时，里面真正要用的文件名：下载后解出并删掉外层包
    channel: int | None = None  # 多声道录音只取这一个声道；None = 各声道平均（语音的默认做法）。设了它的源会按 long_noise_chunk_sec 切成段
    min_bandwidth_hz: int = 0  # 有效带宽低于它的条目丢弃（drop_reason=narrowband）；0 = 不过滤。给噪声设 10000，避免带宽被限制的噪声教坏模型
    meta_csv: str = ""  # 包内一个 CSV 的路径，按 meta_cols 给出每个文件的上游切分和分组（地点）；空 = 不用
    meta_cols: tuple[str, str, str] = ("", "", "")  # (文件名列, 切分列, 分组列)；分组写进 speaker 字段
    note: str = ""

    @property
    def raw_dir(self) -> Path:
        return RAW / (self.raw_name or self.name)


def _openslr(n: int, *files: str) -> tuple[str, ...]:
    return tuple(f"https://www.openslr.org/resources/{n}/{f}" for f in files)


_DEMAND = ("DKITCHEN DLIVING DWASHING NFIELD NPARK NRIVER OHALLWAY OMEETING OOFFICE "
           "PCAFETER PRESTO PSTATION SPSQUARE STRAFFIC TBUS TCAR TMETRO").split()
_XIPH = "https://media.xiph.org/rnnoise/data/"
_DNS5 = "https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset/noise_fullband/datasets_fullband.noise_fullband."

SOURCES: dict[str, Source] = {s.name: s for s in [
    # ---- 英文语音 ----
    Source("libritts_r", "speech", "en", "CC BY 4.0",
           _openslr(141, "train_clean_100.tar.gz", "train_clean_360.tar.gz",
                    "dev_clean.tar.gz", "test_clean.tar.gz"),
           speaker=r"/(?P<spk>\d+)/\d+/[^/]+\.wav$",
           split=r"(?P<split>train|dev|test)-clean",
           note="24kHz，带宽约 12k（B 级）"),
    Source("vctk", "speech", "en", "CC BY 4.0",
           ("https://datashare.ed.ac.uk/download/DS_10283_3443.zip",), unwrap="VCTK-Corpus-0.92.zip",
           include=r"_mic1\.flac$", speaker=r"/(?P<spk>[ps]\d+)/",  # 110 个说话人：p225…p376 和 s5
           note="48kHz 原生，A 级；下载的是 11.7GB 外层包，会解出内层 zip（约 11.7GB）"),
    Source("ears", "speech", "en", "CC BY-NC 4.0（待核实）",
           tuple(f"https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p{i:03d}.zip"
                 for i in range(1, 108)),
           speaker=r"(?P<spk>p\d{3})/", note="48kHz 表现力语音，A 级"),
    # ---- 中文语音 ----
    Source("aishell1", "speech", "zh", "Apache 2.0",
           _openslr(33, "data_aishell.tgz"),
           speaker=r"(?P<spk>S\d{4})/[^/]+\.wav$",
           split=r"/(?P<split>train|dev|test)/S\d{4}/",
           note="16kHz，C 级；内层 wav 是按说话人的嵌套 tar.gz"),
    Source("aishell3", "speech", "zh", "Apache 2.0",
           _openslr(93, "data_aishell3.tgz"),
           speaker=r"(?P<spk>SSB\d{4})/", note="44.1kHz，A 级"),
    # ---- RNNoise 官方 ----
    Source("rnnoise_noise", "noise", "-", "见 Xiph 说明（待核实）",
           (_XIPH + "background_noise_v2.sw", _XIPH + "foreground_noise_v2.sw",
            _XIPH + "contrib_noise.sw"), kind="raw"),
    # ---- 噪声 ----
    Source("demand", "noise", "-", "CC BY 4.0",
           tuple(f"https://zenodo.org/records/1227121/files/{s}_48k.zip" for s in _DEMAND),
           include=r"ch01\.wav$", note="16 路麦克风阵列只取 ch01；VoiceBank-DEMAND 评测集也用了它，注意泄漏"),
    # ---- DNS Challenge 5 噪声：AudioSet（CC BY 4.0）+ Freesound（只选了 CC0），约 150 类，已用语音活动检测去掉含人声的片段 ----
    Source("dns5_noise", "noise", "-", "AudioSet CC BY 4.0 / Freesound CC0",
           tuple(_DNS5 + f"{k}_{i:03d}.tar.bz2" for k, n in (("audioset", 7), ("freesound", 2)) for i in range(n)),
           min_bandwidth_hz=10000,
           note="压缩包共约 39GB（bz2，解压慢）；AudioSet 片段来自 YouTube（有损压缩，原为 44.1kHz），带宽 < 10 kHz 的丢弃；每个文件按哈希独立切分"),
    # ---- Expresso：4 位演员的表演性语音（朗读 + 即兴对话，26 种风格，含耳语/大笑/大喊/非语言发声）----
    Source("expresso", "speech", "en", "CC BY-NC 4.0",
           ("https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar",),
           include=r"audio_48khz/(read|conversational)/.*\.wav$", channel=0,
           speaker=r"audio_48khz/(?:read|conversational)/(?P<spk>ex\d{2})",
           note="38GB 单个 tar；48kHz 24 位；即兴对话是双声道（一人一声道，互相有串音），只取声道 0，speaker 取声道 0 的演员；"
                "对话文件长达 20 分钟，按 long_noise_chunk_sec 切段"),
    # ---- 混响：按房间切分（speaker 字段在这里存房间/场地 id），同一个房间的 RIR 不能跨集合 ----
    Source("rirs_simulated", "rir", "-", "Apache 2.0",
           _openslr(28, "rirs_noises.zip"), raw_name="rirs_noises",
           include=r"simulated_rirs/.*\.wav$", speaker=r"simulated_rirs/(?P<spk>\w+room/Room\d+)/",
           note="600 个房间 x 100 条；Room 编号在三种房型间不唯一，所以键里带房型目录"),
    Source("rirs_real", "rir", "-", "Apache 2.0", raw_name="rirs_noises",
           include=r"real_rirs_isotropic_noises/(?!.*_noise_)[^/]*\.wav$",
           speaker=r"real_rirs_isotropic_noises/(?P<spk>.*?)"
                   r"(?:_(?:\d+|hfrp|hhp|far|near|angla|anglb|imp\d*|rev|[pm]?\d+[lr]))+\.wav$",
           note="排除目录里 92 个 *_noise_* 各向同性噪声录音（它们不是 RIR）；房间 id = 去掉位置/角度后缀的文件名"),
    # ---- WHAM! 48kHz 噪声：旧金山湾区咖啡馆/餐厅/酒吧/公园的双耳录音，已去掉可听懂的语音；44 个地点，78.5 小时 ----
    Source("wham_noise", "noise", "-", "CC BY-NC 4.0（仅限非商用）",
           ("https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/high_res_wham.zip",),
           include=r"high_res_wham/audio/.*\.wav$", channel=0, min_bandwidth_hz=10000,
           meta_csv="high_res_wham/high_res_metadata.csv", meta_cols=("Filename", "WHAM! Split", "Location ID"),
           note="48kHz 24 位双声道，只取左声道（下混会在高频造成梳状滤波）；切分用上游的按地点划分（地点不跨集合），speaker 字段存地点 id；"
                "最大的文件 49 分钟，按 long_noise_chunk_sec 切段后处理"),
]}
