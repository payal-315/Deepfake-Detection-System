import os
import glob
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from sklearn.metrics import accuracy_score

torch.backends.cudnn.benchmark = True  # ✅ speedup

# ==============================
# CONFIG
# ==============================
VIDEO_DIR1 = r"H:\Deepfake_T\MAVOS-DD\english"
VIDEO_DIR2 = r"H:\Deepfake_T\MAVOS-DD\hindi"
VIDEO_DIR3 = r"H:\Deepfake_T\MAVOS-DD\arabic"
VIDEO_DIR4 = r"H:\Deepfake_T\MAVOS-DD\german"
VIDEO_DIR5 = r"H:\Deepfake_T\MAVOS-DD\mandarin"
VIDEO_DIR6 = r"H:\Deepfake_T\MAVOS-DD\romanian"
VIDEO_DIR7 = r"H:\Deepfake_T\MAVOS-DD\russian"
VIDEO_DIR8 = r"H:\Deepfake_T\MAVOS-DD\spanish"

IMG_SIZE = 225
SAMPLE_RATE = 16000
VIDEO_FRAMES = 30
VIDEO_STRIDE = 10
AUDIO_FRAMES = 50
AUDIO_STRIDE = 15
BATCH_SIZE = 64   # ✅ leverage GPU
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# DATASET
# ==============================
import librosa
import subprocess
import os




import cv2
import torch
import torchaudio
import numpy as np
import gc

IMG_SIZE = 225
VIDEO_FRAMES = 30
AUDIO_FRAMES = 40
STRIDE = 15
SAMPLE_RATE = 16000



def predictvideo(filepath, model, device="cuda"):
    cap = cv2.VideoCapture(filepath)
    fps_val = cap.get(cv2.CAP_PROP_FPS) or 0.0
    all_frames = []

    # ---- Load all video frames ----
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame)

    cap.release()

    all_frames = np.array(all_frames)  # [N, H, W, C]
    duration_sec = len(all_frames) / fps_val if fps_val > 0 else None
    # ---- Load full audio ----
    waveform, sr = load_audio_from_mp4(filepath, target_sr=SAMPLE_RATE, duration_sec=duration_sec)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=256,
        win_length=200,
        hop_length=80,
        n_mels=40
    )(waveform)
    mel = torchaudio.transforms.AmplitudeToDB()(mel)
    mel = mel.squeeze(0).transpose(0, 1)  # [Time, 40]

    # ---- Predictions ----
    results = []
    num_frames = len(all_frames)
    mel_factor = mel.shape[0] / num_frames  # mel frames per video frame

    for start in range(0, num_frames - VIDEO_FRAMES + 1, STRIDE):
        # ---- VIDEO CLIP ----
        clip = all_frames[start:start+VIDEO_FRAMES]
        clip = torch.tensor(clip, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0  # [C, T, H, W]
        clip = clip.unsqueeze(0).to(device)  # [1, C, T, H, W]

        # ---- AUDIO CLIP ----
        audio_start = int(start * mel_factor)
        audio_end   = int((start + VIDEO_FRAMES) * mel_factor)
        audio_patch = mel[audio_start:audio_end]

        # Pad / crop to AUDIO_FRAMES (40)
        if audio_patch.shape[0] < AUDIO_FRAMES:
            pad = torch.zeros(AUDIO_FRAMES - audio_patch.shape[0], mel.shape[1])
            audio_patch = torch.cat([audio_patch, pad], dim=0)
        elif audio_patch.shape[0] > AUDIO_FRAMES:
            audio_patch = audio_patch[:AUDIO_FRAMES]

        audio_tensor = audio_patch.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 40, 40]

        # ---- MODEL PREDICTION ----
        with torch.no_grad():
            output = model(clip, audio_tensor)  # <-- adjust if your forward takes 2 inputs
            pred = torch.softmax(output, dim=1).cpu().numpy()
            results.append(pred)

        # ---- Free memory ----
        del clip, audio_tensor, output
        torch.cuda.empty_cache()
        gc.collect()

    return np.array(results),fps_val,duration_sec  # predictions for all clips



def predictvideoonly2(filepath, model, device="cuda"):
    cap = cv2.VideoCapture(filepath)
    fps_val = cap.get(cv2.CAP_PROP_FPS) or 0.0
    all_frames = []

    # ---- Load all video frames ----
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame)

    cap.release()

    all_frames = np.array(all_frames)  # [N, H, W, C]
    duration_sec = len(all_frames) / fps_val if fps_val > 0 else None
    # ---- Load full audio ----

    if duration_sec is None:
            duration_sec = 1.0

    samples = int(16000 * duration_sec)
    waveform, sr = torch.zeros(samples, dtype=torch.float32), 16000
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=256,
        win_length=200,
        hop_length=80,
        n_mels=40
    )(waveform)
    mel = torchaudio.transforms.AmplitudeToDB()(mel)
    mel = mel.squeeze(0).transpose(0, 1)  # [Time, 40]

    # ---- Predictions ----
    results = []
    num_frames = len(all_frames)
    mel_factor = mel.shape[0] / num_frames  # mel frames per video frame

    for start in range(0, num_frames - VIDEO_FRAMES + 1, STRIDE):
        # ---- VIDEO CLIP ----
        clip = all_frames[start:start+VIDEO_FRAMES]
        clip = torch.tensor(clip, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0  # [C, T, H, W]
        clip = clip.unsqueeze(0).to(device)  # [1, C, T, H, W]

        # ---- AUDIO CLIP ----
        audio_start = int(start * mel_factor)
        audio_end   = int((start + VIDEO_FRAMES) * mel_factor)
        audio_patch = mel[audio_start:audio_end]

        # Pad / crop to AUDIO_FRAMES (40)
        if audio_patch.shape[0] < AUDIO_FRAMES:
            pad = torch.zeros(AUDIO_FRAMES - audio_patch.shape[0], mel.shape[1])
            audio_patch = torch.cat([audio_patch, pad], dim=0)
        elif audio_patch.shape[0] > AUDIO_FRAMES:
            audio_patch = audio_patch[:AUDIO_FRAMES]

        audio_tensor = audio_patch.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 40, 40]

        # ---- MODEL PREDICTION ----
        with torch.no_grad():
            output = model(clip, audio_tensor)  # <-- adjust if your forward takes 2 inputs
            pred = torch.softmax(output, dim=1).cpu().numpy()
            results.append(pred)

        # ---- Free memory ----
        del clip, audio_tensor, output
        torch.cuda.empty_cache()
        gc.collect()

    return np.array(results),fps_val,duration_sec  # predictions for all clips


def load_audio_from_mp4(filepath, target_sr=16000, duration_sec=None):
    wav_path = filepath.replace(".mp4", "_temp.wav")

    # Extract audio using ffmpeg
    if not os.path.exists(wav_path):
        subprocess.run([
            r"C:\ffmpeg\ffmpeg-2025-08-25-git-1b62f9d3ae-full_build\bin\ffmpeg.exe", "-y", "-i", filepath, "-ar", str(target_sr), "-ac", "1", wav_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.exists(wav_path):
        print(f"[WARN] No audio detected in {filepath}. Using dummy audio.")
        if duration_sec is None:
            duration_sec = 1.0
        # create 1 second of silence
        samples = int(target_sr * duration_sec)
        return torch.zeros(samples, dtype=torch.float32), target_sr
    # Load with librosa
    audio_array, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    return torch.tensor(audio_array, dtype=torch.float32), sr

def extract_face_region(frame, img_size=224, padding= 15):
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load OpenCV face detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        # Get min/max coordinates covering all faces
        x1 = min([x for (x, y, w, h) in faces])
        y1 = min([y for (x, y, w, h) in faces])
        x2 = max([x + w for (x, y, w, h) in faces])
        y2 = max([y + h for (x, y, w, h) in faces])

        # Apply padding (stronger for height as you had earlier)
        padx = padding * 3
        pady = padding * 4

        x1 = max(0, x1 - padx)
        y1 = max(0, y1 - pady)
        x2 = min(frame.shape[1], x2 + padx)
        y2 = min(frame.shape[0], y2 + pady)

        face_region = frame[y1:y2, x1:x2]
        bbox = (x1, y1, x2, y2)
    else:
        # If no face detected, fallback
        face_region = frame
        bbox = None

    # Resize to desired size
    face_region = cv2.resize(face_region, (img_size, img_size))
    return face_region, bbox


# def extract_face_region(frame, img_size=224, padding=40):
#     # Convert to grayscale for detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Load OpenCV face detector (Haar Cascade)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     if len(faces) > 0:
#         # Take the largest detected face
#         (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])

#         # Add padding around face (portrait-like crop)
#         x1 = max(0, x - padding)
#         y1 = max(0, y - padding)
#         x2 = min(frame.shape[1], x + w + padding)
#         y2 = min(frame.shape[0], y + h + padding)

#         face_region = frame[y1:y2, x1:x2]
#     else:
#         # If no face detected, use the whole frame fallback
#         face_region = frame

#     # Resize to desired size
#     face_region = cv2.resize(face_region, (img_size, img_size))
#     return face_region

def fpredictvideo(filepath, model, device="cuda", save_with_boxes=None):
    cap = cv2.VideoCapture(filepath)
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    if fps_val <= 1:   # fallback if invalid
        fps_val = 25.0   # default fps

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if save_with_boxes is None:
        base, ext = os.path.splitext(filepath)
        save_with_boxes = f"{base}_box{ext}"
    # Video writer for saving annotated output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # or *"avc1" for mp4
    out = cv2.VideoWriter(save_with_boxes, fourcc, fps_val, (width, height))

    all_frames = []

    # ---- Load all video frames ----
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_region, bbox = extract_face_region(frame, img_size=224, padding=50)

        # Draw bounding box if found
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        out.write(frame)  # save annotated frame

        # prepare cropped frame for model
        frame_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        all_frames.append(frame_rgb)

    out.release()
    cap.release()

    all_frames = np.array(all_frames)  # [N, H, W, C]
    duration_sec = len(all_frames) / fps_val if fps_val > 0 else None
    # ---- Load full audio ----
    waveform, sr = load_audio_from_mp4(filepath, target_sr=SAMPLE_RATE, duration_sec=duration_sec)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=256,
        win_length=200,
        hop_length=80,
        n_mels=40
    )(waveform)
    mel = torchaudio.transforms.AmplitudeToDB()(mel)
    mel = mel.squeeze(0).transpose(0, 1)  # [Time, 40]

    # ---- Predictions ----
    results = []
    num_frames = len(all_frames)
    mel_factor = mel.shape[0] / num_frames  # mel frames per video frame

    for start in range(0, num_frames - VIDEO_FRAMES + 1, STRIDE):
        # ---- VIDEO CLIP ----
        clip = all_frames[start:start+VIDEO_FRAMES]
        clip = torch.tensor(clip, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0  # [C, T, H, W]
        clip = clip.unsqueeze(0).to(device)  # [1, C, T, H, W]

        # ---- AUDIO CLIP ----
        audio_start = int(start * mel_factor)
        audio_end   = int((start + VIDEO_FRAMES) * mel_factor)
        audio_patch = mel[audio_start:audio_end]

        # Pad / crop to AUDIO_FRAMES (40)
        if audio_patch.shape[0] < AUDIO_FRAMES:
            pad = torch.zeros(AUDIO_FRAMES - audio_patch.shape[0], mel.shape[1])
            audio_patch = torch.cat([audio_patch, pad], dim=0)
        elif audio_patch.shape[0] > AUDIO_FRAMES:
            audio_patch = audio_patch[:AUDIO_FRAMES]

        audio_tensor = audio_patch.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 40, 40]

        # ---- MODEL PREDICTION ----
        with torch.no_grad():
            output = model(clip, audio_tensor)  # <-- adjust if your forward takes 2 inputs
            pred = torch.softmax(output, dim=1).cpu().numpy()
            results.append(pred)

        # ---- Free memory ----
        del clip, audio_tensor, output
        torch.cuda.empty_cache()
        gc.collect()

    os.remove(filepath)
    tmp_path = save_with_boxes.replace("box.mp4", "fixed.mp4")
   
    subprocess.run([
        r"C:\ffmpeg\ffmpeg-2025-08-25-git-1b62f9d3ae-full_build\bin\ffmpeg.exe", "-y",
        "-i", save_with_boxes,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",   # ✅ re-encode video to H.264
        "-c:a", "aac", "-b:a", "128k",                       # ✅ ensure AAC audio (browser safe)
        "-movflags", "+faststart",                           # ✅ put moov atom at start
        tmp_path
    ])
    return np.array(results),fps_val,duration_sec,tmp_path

# MODEL (same as yours, verified)
# ==============================
class Video3DCNN(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2)),

            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2)),

            nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).unsqueeze(1)  # [B, 1, embed_dim]

class AudioEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x):
        B, T, time, freq = x.shape
        x = x.view(B * T, 1, time, freq)
        x = self.cnn(x)
        x = x.view(B, T, -1).mean(1)
        return self.fc(x).unsqueeze(1)  # [B, 1, embed_dim]

class MultiModalTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2, num_classes=2):
        super().__init__()
        self.video_encoder = Video3DCNN(embed_dim)
        self.audio_encoder = AudioEncoder(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, video, audio):
        v_feat = self.video_encoder(video)
        a_feat = self.audio_encoder(audio)
        seq = torch.cat([v_feat, a_feat], dim=1)
        B = seq.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls_tokens, seq], dim=1)
        out = self.transformer(seq)
        cls_out = out[:, 0]
        return self.fc(cls_out)

# ==============================

