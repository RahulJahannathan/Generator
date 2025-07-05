import cv2
import librosa
import numpy as np
import os
import pickle

# -------- CONFIG --------------
video_path = "input_video.mp4"
audio_path = "audio.wav"
output_pickle = "mfcc_frame_dict.pkl"
sr = 22050
hop_length = 512  # ~23ms
n_mfcc = 13
# ------------------------------

# Step 1: Extract MFCCs
y, _ = librosa.load(audio_path, sr=sr)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
mfcc = mfcc.T  # shape: (frames, 13)

# Step 2: Extract frames from video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
video_hop_duration = hop_length / sr  # ~23ms
mfcc_frame_map = {}
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret or frame_idx >= len(mfcc):
        break
    mfcc_vector = mfcc[frame_idx]
    key = tuple(np.round(mfcc_vector, 1))  # Reduce precision to hash it
    if key not in mfcc_frame_map:
        mfcc_frame_map[key] = frame
    frame_idx += 1
cap.release()

with open(output_pickle, "wb") as f:
    pickle.dump(mfcc_frame_map, f)

print(f"✅ Saved {len(mfcc_frame_map)} unique MFCC-based frames.")
