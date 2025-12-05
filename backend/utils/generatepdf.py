from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import os
import pdfkit
from PIL import Image
from PIL.ExifTags import TAGS

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np



import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")   # Use Anti-Grain Geometry backend (no GUI)

import matplotlib.pyplot as plt


import subprocess


basedir =  "E:/intern_project/CID/DeepfakeDetection/app1/backend"


def extract_thumbnail(video_path, output_path="uploads/thumbnail/thumbnail.png"):
    thumb_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-map", "0:v",  # map video stream
        "-map", "0:m:attach?",  # try attachments (cover art / thumbnail)
        "-frames:v", "1",
        output_path
    ]
    result = subprocess.run(thumb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"✅ Extracted embedded thumbnail -> {output_path}")
        return output_path

    # 2. If no embedded thumbnail, fall back to first frame
    frame_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", "select=eq(n\\,0)",  # first frame only
        "-q:v", "3",                # quality
        output_path
    ]
    subprocess.run(frame_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"✅ Extracted first frame as thumbnail -> {output_path}")
        return output_path
    else:
        raise RuntimeError("❌ Could not extract thumbnail")



def generate_image_pdf(filepath: str, result: dict, out_path: str = "uploads/pdf/forensic_report.pdf",grad_full="",grad_face=""):
    """
    Generate a forensic PDF from an HTML template using pdfkit + wkhtmltopdf.
    """
    # Jinja2 setup
    abs_out_path = f"{basedir}/{out_path}"

    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("image_report.html")
    image_abs_path = os.path.abspath(filepath)
    if grad_full:
        grad_full = f"{basedir}/{grad_full}"
    if grad_face:
        grad_face = f"{basedir}/{grad_face}"   

    file_stat = os.stat(image_abs_path)
    
    try:
        with Image.open(image_abs_path) as img:
            width, height = img.size
            image_mode = img.mode
            image_format = img.format
    except Exception:
        width = height = 0
        image_mode = image_format = "Unknown"

    file_details = {
        "size_bytes": file_stat.st_size,
        "size_kb": round(file_stat.st_size / 1024, 2),
        "created": datetime.fromtimestamp(file_stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
        "modified": datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "accessed": datetime.fromtimestamp(file_stat.st_atime).strftime("%Y-%m-%d %H:%M:%S"),
        "width": width,
        "height": height,
        "mode": image_mode,
        "format": image_format,
        "absolute_path": image_abs_path
    }
    context = {
        "filename": result.get("filename", "unknown.jpg"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_deepfake": result.get("is_deepfake", False),
        "confidence": f"{result.get('confidence', 0.0):.2f}",
        "image_path":image_abs_path,
        "file_details": file_details,
        "gradcam_full_path": grad_full,
        "gradcam_face_path": grad_face
    }

    # Render HTML
    html_content = template.render(context)

    options = {
        "enable-local-file-access": ""
    }
    try:
        pdfkit.from_string(html_content, abs_out_path, options=options)
        print("Saving to:", abs_out_path)
    except Exception as e:
        print("[ERROR]", e)

    return out_path


def generate_waveform(audio_path, out_path):
    """Generate waveform plot for given audio file"""
    
    y, sr = librosa.load(audio_path, sr=None)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(os.path.basename(audio_path))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
  
    plt.close()
    return out_path

def generate_audio_pdf(file_id,reference_path, test_path, result: dict, out_path="uploads/pdf/audio_report.pdf"):
    abs_out_path = f"{basedir}/{out_path}"
    
    # Generate waveform images

    ref_waveform = generate_waveform(reference_path, f"{basedir}/uploads/waveforms/{file_id}_reference.png")
    test_waveform = generate_waveform(test_path, f"{basedir}/uploads/waveforms/{file_id}_test.png")

    # Jinja2 setup
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("audio_report.html")

    # Context data
    context = {
        "reference_filename": result.get("reference_filename", "reference.wav"),
        "test_filename": result.get("test_filename", "test.wav"),
        "similarity": f"{result.get('similarity', 0.0):.2f}",
        "probability": f"{result.get('probability', 0.0):.2f}",
        "verdict": result.get("verdict", "Unknown"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reference_path": os.path.abspath(reference_path),
        "test_path": os.path.abspath(test_path),
        "ref_url":reference_path,
        "test_url":test_path,
        "reference_waveform": os.path.abspath(ref_waveform),
        "test_waveform": os.path.abspath(test_waveform),
    }

    # Render HTML
    html_content = template.render(context)

    # Ensure output folder exists
    options = {"enable-local-file-access": ""}
    pdfkit.from_string(html_content, abs_out_path, options=options)

    return out_path


def generate_video_pdf(filepath: str, result: dict, out_path: str = "uploads/pdf/video_report.pdf", base_url="http://127.0.0.1:8000/"):
    """
    Generate a forensic PDF for video analysis.
    """
    abs_out_path = f"{basedir}/{out_path}"
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("video_report.html")

    # Generate per-frame score graph
    per_frame_scores = result["details"].get("per_frame_scores", [])
    graph_path = f"uploads/graphs/{result['id']}_perframe.png"
    abs_graph_path = f"{basedir}/{graph_path}"

    if per_frame_scores:
        fps = result["details"].get("fps", 25)
        x = np.array([i / fps for i in range(len(per_frame_scores))])
        y = np.array(per_frame_scores)

        # Add top/bottom padding
        y_scaled = y * 0.95 + 0.025

        # Determine color per frame
        colors = ["red" if v > 0.6 else "yellow" if v > 0.35 else "green" for v in y_scaled]

        plt.figure(figsize=(16, 9))  # wider for landscape

        for i in range(len(x) - 1):
            plt.plot(x[i:i+2], y_scaled[i:i+2], color=colors[i], linewidth=1.5)
            plt.fill_between(x[i:i+2], 0, y_scaled[i:i+2], color=colors[i], alpha=0.2)

        plt.title("Per-frame Deepfake Confidence", fontsize=18)
        plt.xlabel("Time (seconds)", fontsize=14)
        plt.ylabel("Confidence Score", fontsize=14)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        os.makedirs(os.path.dirname(abs_graph_path), exist_ok=True)
        plt.savefig(abs_graph_path, dpi=200)
        plt.close()


    # Use first frame as thumbnail (placeholder if you don’t extract)
    thumbnail_path = extract_thumbnail(filepath,f"uploads/thumbnail/{result['id']}.png") # TODO: replace with real extraction

    abs_thumbnail_path = f"{basedir}/{thumbnail_path}"

    # Context data
    context = {
        "filename": result.get("filename", "unknown.mp4"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_deepfake": result.get("is_deepfake", False),
        "confidence": f"{result.get('confidence', 0.0):.2f}",
        "video_url": result["details"].get("video_url", ""),
        "frames_analyzed": result["details"].get("frames_analyzed", 0),
        "fps": result["details"].get("fps", 25),
        "duration_seconds": result["details"].get("duration_seconds", 0),
        "audio_sync_score": f"{result['details'].get('audio_sync_score', 0.0):.2f}",
        "temporal_consistency": f"{result['details'].get('temporal_consistency', 0.0):.2f}",
        "perframe_graph": abs_graph_path,
        "thumbnail_path": abs_thumbnail_path,
        "absolute_path": os.path.abspath(filepath),
        "base_url": base_url,
    }

    # Render HTML
    html_content = template.render(context)

    options = {
        "enable-local-file-access": "",
        "enable-external-links": "",
    }


    pdfkit.from_string(html_content, abs_out_path, options=options)
    return out_path
