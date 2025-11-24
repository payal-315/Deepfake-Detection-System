import os
import random
import math
import json
from dataclasses import asdict, dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T

# Optional deps
try:
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    HAS_SPEECHBRAIN = True
except Exception:
    HAS_SPEECHBRAIN = False

from sklearn import metrics
import torchaudio.transforms as T

from pydub import AudioSegment
import io
# -----------------------
# CONFIG
import os
import glob
from sklearn.model_selection import train_test_split

# -----------------------
@dataclass
class Config:
    dataset_dir: str = r"D:\deepfake\audiodata"  # root folder with speaker folders
    sample_rate: int = 16000
    n_mels: int = 64
    n_lfcc: int = 40
    n_fft: int = 1024
    hop_length: int = 256
    max_frames: int = 512                # final time frames after pad/crop
    batch_size: int = 32
    num_workers: int = 2
    lr: float = 1e-4
    epochs: int = 10
    margin: float = 1.0                  # for contrastive loss
    specaug_time_mask_p: int = 40        # width (frames)
    specaug_freq_mask_p: int = 8         # width (mels/lfcc)
    balance_pairs_per_speaker: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./checkpoints_forensic"
    resume_from: Optional[str] = None     # path to .pth to resume (optional)
    # Progress throttle (optional): process at most N batches per epoch (set None for full)
    max_batches_per_epoch: Optional[int] = 1000

CFG = Config()
os.makedirs(CFG.save_dir, exist_ok=True)
random.seed(CFG.seed)
torch.manual_seed(CFG.seed)

# -----------------------
# AUDIO / FEAT PIPELINE
# -----------------------
def safe_load_audio(path: str, target_sr: int) -> Optional[torch.Tensor]:
    """Return mono waveform [1, T] at target_sr or None if unreadable."""
    try:
        # First try torchaudio
        wav, sr = torchaudio.load(path)
    except Exception as e:
        try:
            # Fallback: use pydub (supports mp3, m4a, etc.)
            audio = AudioSegment.from_file(path)
            sr = audio.frame_rate
            samples = audio.get_array_of_samples()
            wav = torch.tensor(samples, dtype=torch.float32).view(-1, audio.channels).T
        except Exception as e2:
            print(f"[WARN] Could not read: {path} | {e2}")
            return None

    # to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # resample
    if sr != target_sr:
        wav = T.Resample(orig_freq=sr, new_freq=target_sr)(wav)

    return wav  # [1, T]

import torchvision.models as models

class AudioEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # remove FC
        self.fc = nn.Linear(base.fc.in_features, embed_dim)

    def forward(self, x):  # x: [B, 1, n_mels, T]
        feat = self.feature_extractor(x)
        feat = feat.flatten(1)
        return self.fc(feat)

class AudioClassifier(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.encoder = AudioEncoder(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        emb = self.encoder(x)
        return self.classifier(emb)

class DeepfakeClassifier(nn.Module):
    def __init__(self, encoder, embed_dim=256, hidden_dim=128):
        super().__init__()
        self.encoder = encoder
        # feature vector size = e1 + e2 + |e1-e2| + (e1*e2)
        fusion_dim = embed_dim * 4
        self.lstm = nn.LSTM(fusion_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim*2, 1)
        self.fc = nn.Linear(hidden_dim*2, 1)

    def forward(self, x1, x2):
        e1 = self.encoder(x1)   # [B, embed_dim]
        e2 = self.encoder(x2)   # [B, embed_dim]

        diff = torch.abs(e1 - e2)
        mult = e1 * e2
        fusion = torch.cat([e1, e2, diff, mult], dim=1)  # [B, embed_dim*4]

        fusion = fusion.unsqueeze(1)  # [B, 1, fusion_dim]
        out, _ = self.lstm(fusion)
        attn_w = torch.softmax(self.attn(out), dim=1)
        context = (out * attn_w).sum(dim=1)

        return self.fc(context)


class FeatureExtractor(nn.Module):
    """Create concatenated Mel + LFCC features with optional SpecAugment."""
    def __init__(self, sr: int, n_mels: int, n_lfcc: int, n_fft: int, hop: int,
                 max_frames: int, apply_specaug: bool):
        super().__init__()
        self.mel = T.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
        self.mel_db = T.AmplitudeToDB()
        self.lfcc = T.LFCC(sample_rate=sr, n_filter=n_mels, n_lfcc=n_lfcc, speckwargs={"n_fft": n_fft, "hop_length": hop})
        self.max_frames = max_frames
        self.apply_specaug = apply_specaug
        # SpecAugment
        self.tmask = T.TimeMasking(time_mask_param=CFG.specaug_time_mask_p)
        self.fmask = T.FrequencyMasking(freq_mask_param=CFG.specaug_freq_mask_p)

    def forward(self, wav: torch.Tensor, train_mode: bool) -> torch.Tensor:
        """
        wav: [1, T] mono
        Returns: features [1, F, max_frames] where F = n_mels + n_lfcc
        """
        mel = self.mel(wav)              # [1, n_mels, Tm]
        mel = self.mel_db(mel)
        lfcc = self.lfcc(wav)            # [1, n_lfcc, Tl]

        # Make time dims equal via center-crop or pad to the longer, then to max_frames
        # First, match mel and lfcc time length
        t = min(mel.size(-1), lfcc.size(-1))
        mel = mel[..., :t]
        lfcc = lfcc[..., :t]

        feats = torch.cat([mel, lfcc], dim=1)  # [1, n_mels+n_lfcc, t]

        # Pad / crop to max_frames
        if feats.size(-1) < self.max_frames:
            pad = self.max_frames - feats.size(-1)
            feats = nn.functional.pad(feats, (0, pad))
        else:
            feats = feats[..., :self.max_frames]

        # SpecAugment only during training
        if self.apply_specaug and train_mode:
            feats = self.fmask(feats)
            feats = self.tmask(feats)

        return feats  # [1, F, max_frames]


def collate_pairs(batch, fe: FeatureExtractor):
    """
    Batch: list of (path1, path2, label)
    Returns: x1 [B, 1, F, T], x2 [B, 1, F, T], labels [B]
    """
    x1_list, x2_list, y_list = [], [], []
    for p1, p2, y in batch:
        wav1 = safe_load_audio(p1, CFG.sample_rate)
        wav2 = safe_load_audio(p2, CFG.sample_rate)
        if wav1 is None or wav2 is None:
            # skip this item by not appending; we will handle empty batch above
            continue
        f1 = fe(wav1, train_mode=True)   # [1, F, T]
        f2 = fe(wav2, train_mode=True)
        x1_list.append(f1)
        x2_list.append(f2)
        y_list.append(y)

    if len(x1_list) == 0:
        # create an empty tensor batch to avoid crash; caller will skip
        return None, None, None

    x1 = torch.stack(x1_list, dim=0)  # [B, 1, F, T]
    x2 = torch.stack(x2_list, dim=0)  # [B, 1, F, T]
    y = torch.stack(y_list, dim=0)    # [B]
    return x1, x2, y






class ECAPAEncoder(nn.Module):
    """ECAPA-TDNN expecting input [B, F, T]."""
    def __init__(self, feat_dim: int):
        super().__init__()
        self.enc = ECAPA_TDNN(input_size=feat_dim, lin_neurons=192)

    def forward(self, x):  # x: [B, 1, F, T]
        x = x.squeeze(1)   # [B, F, T]
        return self.enc(x) # [B, 192]

class StrongCNN(nn.Module):
    """Fallback CNN if SpeechBrain not available."""
    def __init__(self, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 192)

    def forward(self, x):  # [B, 1, F, T]
        z = self.net(x)    # [B, 128, 1, 1]
        z = z.flatten(1)
        return self.fc(z)  # [B, 192]

class Siamese(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        if HAS_SPEECHBRAIN:
            self.encoder = ECAPAEncoder(feat_dim)
        else:
            self.encoder = StrongCNN(feat_dim)

    def embed(self, x):
        return self.encoder(x)  # [B, 192]

    def forward(self, x1, x2):
        e1 = self.embed(x1)
        e2 = self.embed(x2)
        return e1, e2  # embeddings

