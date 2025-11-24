
from __future__ import annotations
import argparse
import math
import os
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image, ImageChops, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tvm


def to_tensor(img: Image.Image) -> torch.Tensor:
    return T.ToTensor()(img)

class SRMFilter(nn.Module):
    def __init__(self):
        super().__init__()
        k1 = np.array([[0,0,0,0,0], [0,-1,2,-1,0], [0,2,-4,2,0], [0,-1,2,-1,0], [0,0,0,0,0]], dtype=np.float32)
        k2 = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]], dtype=np.float32)
        k3 = np.array([[0,0,0,0,0],[0,1,-2,1,0],[0,-2,4,-2,0],[0,1,-2,1,0],[0,0,0,0,0]], dtype=np.float32)
        k = np.stack([k1,k2,k3])[:,None,:,:]  # (3,1,5,5)
        k = torch.from_numpy(k)
        self.register_buffer('weight', k)
        self.pad = nn.ReflectionPad2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        y = F.conv2d(x, self.weight)
        y = (y - y.amin(dim=(2,3), keepdim=True)) / (y.amax(dim=(2,3), keepdim=True) - y.amin(dim=(2,3), keepdim=True) + 1e-6)
        return y

import numpy as np
from PIL import Image
import insightface

class FaceCropper:
    def __init__(self, device: str = "cpu"):
        self.face_model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.face_model.prepare(ctx_id=0 if device == "cuda" else -1)

    def crop_largest_face(self, img: Image.Image) -> Image.Image:
        """Crop the largest face from a PIL image. Returns cropped face or original image if no face."""
        img_np = np.array(img.convert("RGB"))[:, :, ::-1]  # RGB → BGR
        faces = self.face_model.get(img_np)

        if len(faces) > 0:
            # pick largest face
            areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
            i = np.argmax(areas)
            x1, y1, x2, y2 = [int(v) for v in faces[i].bbox]
            return img.crop((x1, y1, x2, y2))

        return img  # fallback: no face found, return original

import numpy as np
from PIL import Image
import insightface

class FaceCropper:
    def __init__(self, device: str = "cpu"):
        self.face_model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.face_model.prepare(ctx_id=0 if device == "cuda" else -1)

    def crop_largest_face(self, img: Image.Image) -> Image.Image:
        """Crop the largest face from a PIL image. Returns cropped face or original image if no face."""
        img_np = np.array(img.convert("RGB"))[:, :, ::-1]  # RGB → BGR
        faces = self.face_model.get(img_np)

        if len(faces) > 0:
            # pick largest face
            areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
            i = np.argmax(areas)
            x1, y1, x2, y2 = [int(v) for v in faces[i].bbox]
            return img.crop((x1+10, y1+10, x2+10, y2+10))

        return img  # fallback: no face found, return original


class ForensicsTransforms:
  
    def __init__(self, size: int = 256, aug_strong: bool = False):
        self.size = size
        self.aug_rgb = T.Compose([
            T.Resize((size,size), interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.1,0.1,0.1,0.05) if aug_strong else T.Lambda(lambda x:x),
        ])
        self.to_tensor = T.ToTensor()
        self.to_gray = T.Compose([T.Resize((size,size), interpolation=Image.BICUBIC), T.Grayscale()])
        self.srm = SRMFilter()
        self.size_hw = (size, size)
        self.aug_strong = aug_strong

    def _error_level_analysis(self, img: Image.Image, quality: int = 90) -> Image.Image:
        base = img.convert('RGB').resize(self.size_hw, Image.BICUBIC)
        tmp = base.copy()
        from io import BytesIO
        buf = BytesIO()
        tmp.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        comp = Image.open(buf)
        ela = ImageChops.difference(base, comp)
        extrema = ela.getextrema()
        scale = 255.0 / max(1, max([ex[1] for ex in extrema]))
        ela = Image.eval(ela, lambda px: int(px*scale))
        ela = ela.convert('L')
        return ela

    def _fft_magnitude(self, img: Image.Image) -> Image.Image:
        g = self.to_gray(img)
        arr = np.array(g).astype(np.float32) / 255.0
        f = np.fft.fft2(arr)
        fshift = np.fft.fftshift(f)
        mag = np.log(1 + np.abs(fshift))
        mag = mag / (mag.max() + 1e-6)
        mag = (mag*255).astype(np.uint8)
        return Image.fromarray(mag)

    def __call__(self, img: Image.Image):
        rgb = self.aug_rgb(img.convert('RGB'))
        rgb = self.to_tensor(rgb)
        gray = self.to_gray(img)
        gray_t = self.to_tensor(gray)  # 1xHxW
        noise = self.srm(gray_t.unsqueeze(0)).squeeze(0)  # 3xHxW
        freq = self._fft_magnitude(img)
        freq_t = self.to_tensor(freq)  # 1xHxW
        ela = self._error_level_analysis(img)
        ela_t = self.to_tensor(ela)  # 1xHxW
        return {
            'rgb': rgb,
            'noise': noise,
            'freq': freq_t,
            'ela': ela_t,
        }
    
class BranchEncoder(nn.Module):
    def __init__(self, in_ch: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class RGBEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        m = tvm.resnet50(weights=None)
        m.fc = nn.Identity()
        self.backbone = m
        self.proj = nn.Linear(2048, out_dim)
    def forward(self, x):
        x = self.backbone(x)
        x = self.proj(x)
        return x
import timm
class XRGBEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        m = timm.create_model("xception", pretrained=True, num_classes=0)  
        # num_classes=0 → removes classifier, outputs feature vector directly
        self.backbone = m
        in_features = m.num_features  # feature dimension of Xception (usually 2048)
        
        self.proj = nn.Linear(in_features, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.proj(x)
        return x

        
class FusionTransformer(nn.Module):
    def __init__(self, emb_dim: int = 256, n_tokens: int = 4, n_heads: int = 4, depth: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim*2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.cls = nn.Parameter(torch.randn(1,1,emb_dim))
        self.head = nn.Linear(emb_dim, 1)

    def forward(self, feats: List[torch.Tensor]):
        x = torch.stack(feats, dim=1) 
        B, T, D = x.shape
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) 
        x = self.encoder(x)
        cls_tok = x[:,0]  # [B, D]
        logit = self.head(cls_tok)
        return logit.squeeze(1)

class MultiCueDetector(nn.Module):
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.rgb_enc = RGBEncoder(out_dim=emb_dim)
        self.noise_enc = BranchEncoder(in_ch=3, out_dim=emb_dim)
        self.freq_enc = BranchEncoder(in_ch=1, out_dim=emb_dim)
        self.ela_enc = BranchEncoder(in_ch=1, out_dim=emb_dim)
        self.fusion = FusionTransformer(emb_dim=emb_dim, n_tokens=4)

    def forward(self, rgb, noise, freq, ela):
        f_rgb = self.rgb_enc(rgb)
        f_noise = self.noise_enc(noise)
        f_freq = self.freq_enc(freq)
        f_ela = self.ela_enc(ela)
        logit = self.fusion([f_rgb, f_noise, f_freq, f_ela])
        return logit, {
            'rgb': f_rgb, 'noise': f_noise, 'freq': f_freq, 'ela': f_ela
        }


class XMultiCueDetector(nn.Module):
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.rgb_enc = XRGBEncoder(out_dim=emb_dim)
        self.noise_enc = BranchEncoder(in_ch=3, out_dim=emb_dim)
        self.freq_enc = BranchEncoder(in_ch=1, out_dim=emb_dim)
        self.ela_enc = BranchEncoder(in_ch=1, out_dim=emb_dim)
        self.fusion = FusionTransformer(emb_dim=emb_dim, n_tokens=4)

    def forward(self, rgb, noise, freq, ela):
        f_rgb = self.rgb_enc(rgb)
        f_noise = self.noise_enc(noise)
        f_freq = self.freq_enc(freq)
        f_ela = self.ela_enc(ela)
        logit = self.fusion([f_rgb, f_noise, f_freq, f_ela])
        return logit, {
            'rgb': f_rgb, 'noise': f_noise, 'freq': f_freq, 'ela': f_ela
        }