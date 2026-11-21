"""数据源注册表。URL 均已用 HEAD 请求核实过可访问（2026-11-20）；许可证请下载前自行复核。"""

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
    raw_name: str = ""  # 多个 Source 共用同一份原始包时指定（如 musan）
    kind: str = "archive"  # archive: tar/zip；raw: 无头 int16 48k（.sw）
    include: str = ""  # 对包内路径的正则，空 = 所有音频
    speaker: str = ""  # 含命名组 spk 的正则；空 = 无说话人信息
    split: str = ""  # 含命名组 split(train|dev|test) 的上游切分；空 = 哈希
    unwrap: str = ""  # 下载到的是外层 zip 时，里面真正要用的文件名：下载后解出并删掉外层包
    note: str = ""

    @property
    def raw_dir(self) -> Path:
        return RAW / (self.raw_name or self.name)


def _openslr(n: int, *files: str) -> tuple[str, ...]:
    return tuple(f"https://www.openslr.org/resources/{n}/{f}" for f in files)


_DEMAND = ("DKITCHEN DLIVING DWASHING NFIELD NPARK NRIVER OHALLWAY OMEETING OOFFICE "
           "PCAFETER PRESTO PSTATION SPSQUARE STRAFFIC TBUS TCAR TMETRO").split()
_XIPH = "https://media.xiph.org/rnnoise/data/"

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
    Source("musan_noise", "noise", "-", "CC BY 4.0（部分条目待核实）",
           _openslr(17, "musan.tar.gz"), raw_name="musan", include=r"/noise/"),
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
]}

SETS: dict[str, list[str]] = {
    "phase1": ["libritts_r", "rnnoise_noise", "demand", "musan_noise", "rirs_simulated", "rirs_real"],
    "phase2": ["aishell1", "aishell3", "vctk", "ears"],
}


def resolve(names: list[str]) -> list[Source]:
    out: list[Source] = []
    for n in names:
        for m in SETS.get(n, [n]):
            if m not in SOURCES:
                raise SystemExit(f"未知数据源: {m}（可用: {', '.join(SOURCES)}；集合: {', '.join(SETS)}）")
            out.append(SOURCES[m])
    return out
