"""
preprocess.py
- sample_frames(video_path, stride_sec, resize)
- detect_scenes(video_path) using PySceneDetect (optional)
- extract_audio_rms(video_path, timestamps)
"""

import cv2, os, numpy as np

def read_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, frames, duration, (w,h)

def sample_frames(path, stride_sec=1.0, resize=224, max_frames=None):
    fps, frames, duration, (w,h) = read_video_info(path)
    cap = cv2.VideoCapture(path)
    stride = max(1, int(round(fps * stride_sec)))
    imgs, ts, idxs = [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if i % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs.append(cv2.resize(rgb, (resize, resize)))
            ts.append(i / fps)
            idxs.append(i)
            if max_frames and len(imgs) >= max_frames:
                break
        i += 1
    cap.release()
    return np.array(imgs), np.array(ts), np.array(idxs), fps, duration

def extract_audio_rms(path, timestamps, sr=16000, hop=512):
    try:
        import librosa
        y, _ = librosa.load(path, sr=sr)
    except Exception as e:
        print("librosa failed:", e); return np.zeros(len(timestamps))
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop).flatten()
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    scores = np.zeros(len(timestamps))
    for i,t in enumerate(timestamps):
        j = np.searchsorted(times, t)
        j = np.clip(j, 0, len(rms)-1)
        scores[i] = float(rms[j])
    if scores.max() > 0:
        scores = (scores - scores.min())/(scores.max()-scores.min()+1e-9)
    return scores

# Optional: simple scene detection fallback (histogram difference)
def simple_scene_detect(path, stride_sec=1.0, resize=160, threshold=0.5):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps * stride_sec)))
    prev = None
    scenes = []
    t = 0.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if i % stride == 0:
            small = cv2.resize(frame, (resize, resize))
            hist = cv2.calcHist([small], [0,1,2], None, [8,8,8], [0,256]*3)
            hist = cv2.normalize(hist, hist).flatten()
            if prev is not None:
                d = cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA)
                if d > threshold:
                    scenes.append(t)
            prev = hist
            t = i / fps
        i += 1
    cap.release()
    return scenes
