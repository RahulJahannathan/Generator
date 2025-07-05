import cv2
import librosa
import numpy as np
import pickle
import os
import moviepy.editor as mp

# -------- CONFIG --------------
audio_path = "audio.wav"
pickle_path = "mfcc_frame_dict.pkl"
output_video = "generated_video23.mp4"
sr = 22050
hop_length = 512
n_mfcc = 13
fps = int(sr / hop_length)  # ~43 fps
# ------------------------------

# Load precomputed MFCC:frame dictionary
with open(pickle_path, "rb") as f:
    mfcc_frame_map = pickle.load(f)

# Extract MFCCs from new audio
y, _ = librosa.load(audio_path, sr=sr)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
mfcc = mfcc.T

# Sample frame for video size
sample_frame = next(iter(mfcc_frame_map.values()))
height, width, _ = sample_frame.shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Stitch video
for vec in mfcc:
    key = tuple(np.round(vec, 1))
    frame = mfcc_frame_map.get(key)
    if frame is not None:
        out.write(frame)

out.release()
print("✅ Generated:", output_video)

video = mp.VideoFileClip("generated_video.mp4")
audio = mp.AudioFileClip("audio.wav").subclip(0, video.duration)  # Trim here
final = video.set_audio(audio)
final.write_videofile("final_output_audio.mp4", codec="libx264", audio_codec="aac")