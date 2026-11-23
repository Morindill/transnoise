"""Registry of data sources. Re-check each license yourself before downloading."""

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
    raw_name: str = ""  # set when several Sources share one raw archive (e.g. rirs_noises)
    kind: str = "archive"  # archive: tar/zip; raw: headerless int16 48 kHz (.sw)
    include: str = ""  # regex on the path inside the archive; empty = all audio
    speaker: str = ""  # regex with a named group spk; empty = no speaker information
    split: str = ""  # regex with a named group split (train|dev|test) for the upstream split; empty = hash
    unwrap: str = ""  # when the download is an outer zip: the file inside that is actually used; extracted after download, outer archive deleted
    channel: int | None = None  # for multichannel recordings take only this channel; None = average the channels (the default for speech). Sources that set it are cut into long_noise_chunk_sec segments
    min_bandwidth_hz: int = 0  # drop entries whose effective bandwidth is below this (drop_reason=narrowband); 0 = no filter. Set to 10000 for noise so band-limited noise does not mislead the model
    meta_csv: str = ""  # path of a CSV inside the archive that gives each file's upstream split and group (location) via meta_cols; empty = unused
    meta_cols: tuple[str, str, str] = ("", "", "")  # (filename column, split column, group column); the group is written to the speaker field
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
    # ---- English speech ----
    Source("libritts_r", "speech", "en", "CC BY 4.0",
           _openslr(141, "train_clean_100.tar.gz", "train_clean_360.tar.gz",
                    "dev_clean.tar.gz", "test_clean.tar.gz"),
           speaker=r"/(?P<spk>\d+)/\d+/[^/]+\.wav$",
           split=r"(?P<split>train|dev|test)-clean",
           note="24 kHz, bandwidth about 12 kHz (tier B)"),
    Source("vctk", "speech", "en", "CC BY 4.0",
           ("https://datashare.ed.ac.uk/download/DS_10283_3443.zip",), unwrap="VCTK-Corpus-0.92.zip",
           include=r"_mic1\.flac$", speaker=r"/(?P<spk>[ps]\d+)/",  # 110 speakers: p225...p376 and s5
           note="Native 48 kHz, tier A; the download is an 11.7 GB outer archive, from which the inner zip (about 11.7 GB) is extracted"),
    Source("ears", "speech", "en", "CC BY-NC 4.0 (to be verified)",
           tuple(f"https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p{i:03d}.zip"
                 for i in range(1, 108)),
           speaker=r"(?P<spk>p\d{3})/", note="48 kHz expressive speech, tier A"),
    # ---- Chinese speech ----
    Source("aishell1", "speech", "zh", "Apache 2.0",
           _openslr(33, "data_aishell.tgz"),
           speaker=r"(?P<spk>S\d{4})/[^/]+\.wav$",
           split=r"/(?P<split>train|dev|test)/S\d{4}/",
           note="16 kHz, tier C; the inner wavs are per-speaker nested tar.gz files"),
    Source("aishell3", "speech", "zh", "Apache 2.0",
           _openslr(93, "data_aishell3.tgz"),
           speaker=r"(?P<spk>SSB\d{4})/", note="44.1 kHz, tier A"),
    # ---- Official RNNoise data ----
    Source("rnnoise_noise", "noise", "-", "See the Xiph notes (to be verified)",
           (_XIPH + "background_noise_v2.sw", _XIPH + "foreground_noise_v2.sw",
            _XIPH + "contrib_noise.sw"), kind="raw"),
    # ---- Noise ----
    Source("demand", "noise", "-", "CC BY 4.0",
           tuple(f"https://zenodo.org/records/1227121/files/{s}_48k.zip" for s in _DEMAND),
           include=r"ch01\.wav$", note="Of the 16-channel microphone array only ch01 is used; the VoiceBank-DEMAND test set also uses DEMAND, so beware of leakage"),
    # ---- DNS Challenge 5 noise: AudioSet (CC BY 4.0) + Freesound (CC0 only), about 150 classes, segments containing speech already removed by voice activity detection ----
    Source("dns5_noise", "noise", "-", "AudioSet CC BY 4.0 / Freesound CC0",
           tuple(_DNS5 + f"{k}_{i:03d}.tar.bz2" for k, n in (("audioset", 7), ("freesound", 2)) for i in range(n)),
           min_bandwidth_hz=10000,
           note="About 39 GB of archives (bz2, slow to decompress); AudioSet clips come from YouTube (lossy, originally 44.1 kHz); clips with bandwidth < 10 kHz are dropped; each file is split by its own hash"),
    # ---- Expresso: expressive speech from 4 actors (read + improvised dialogue, 26 styles, including whisper/laughter/shouting/non-verbal sounds) ----
    Source("expresso", "speech", "en", "CC BY-NC 4.0",
           ("https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar",),
           include=r"audio_48khz/(read|conversational)/.*\.wav$", channel=0,
           speaker=r"audio_48khz/(?:read|conversational)/(?P<spk>ex\d{2})",
           note="A single 38 GB tar; 48 kHz 24-bit; improvised dialogues are two-channel (one speaker per channel, with crosstalk), so only channel 0 is used and speaker is the actor on channel 0; "
                "dialogue files run up to 20 minutes and are cut into long_noise_chunk_sec segments"),
    # ---- Reverb: split by room (the speaker field stores the room/venue id here); RIRs of one room must not cross sets ----
    Source("rirs_simulated", "rir", "-", "Apache 2.0",
           _openslr(28, "rirs_noises.zip"), raw_name="rirs_noises",
           include=r"simulated_rirs/.*\.wav$", speaker=r"simulated_rirs/(?P<spk>\w+room/Room\d+)/",
           note="600 rooms x 100 RIRs; Room numbers are not unique across the three room sizes, so the key includes the room-size directory"),
    Source("rirs_real", "rir", "-", "Apache 2.0", raw_name="rirs_noises",
           include=r"real_rirs_isotropic_noises/(?!.*_noise_)[^/]*\.wav$",
           speaker=r"real_rirs_isotropic_noises/(?P<spk>.*?)"
                   r"(?:_(?:\d+|hfrp|hhp|far|near|angla|anglb|imp\d*|rev|[pm]?\d+[lr]))+\.wav$",
           note="Excludes the 92 *_noise_* isotropic noise recordings in that directory (they are not RIRs); room id = file name without the position/angle suffix"),
    # ---- WHAM! 48 kHz noise: binaural recordings from cafes/restaurants/bars/parks in the San Francisco Bay Area, intelligible speech removed; 44 locations, 78.5 hours ----
    Source("wham_noise", "noise", "-", "CC BY-NC 4.0 (non-commercial only)",
           ("https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/high_res_wham.zip",),
           include=r"high_res_wham/audio/.*\.wav$", channel=0, min_bandwidth_hz=10000,
           meta_csv="high_res_wham/high_res_metadata.csv", meta_cols=("Filename", "WHAM! Split", "Location ID"),
           note="48 kHz 24-bit two-channel, only the left channel is used (downmixing causes comb filtering at high frequencies); the split follows upstream's per-location division (a location never crosses sets) and the speaker field stores the location id; "
                "the largest file is 49 minutes and is processed in long_noise_chunk_sec segments"),
]}
