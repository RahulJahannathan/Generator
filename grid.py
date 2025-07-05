import cv2

# Input videos
video_path_1 = 'result.mp4'
video_path_2 = 'final_output_audio.mp4'

# Output (silent) combined video
output_silent = 'combined_silent.mp4'
final_output = 'final_combined_with_audio.mp4'

# Audio file path (extracted or use one of the video’s audio)
audio_source = 'result.mp4'  # Assumes this video has the correct audio

# Capture videos
cap1 = cv2.VideoCapture(video_path_1)
cap2 = cv2.VideoCapture(video_path_2)

# Read properties
fps = cap1.get(cv2.CAP_PROP_FPS)
width = 640
height = 360
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'

# Output writer (side-by-side = 1280x360)
out = cv2.VideoWriter(output_silent, fourcc, fps, (width * 2, height))

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("One of the videos has ended.")
        break

    frame1 = cv2.resize(frame1, (width, height))
    frame2 = cv2.resize(frame2, (width, height))

    combined = cv2.hconcat([frame1, frame2])
    out.write(combined)

    cv2.imshow('Combined', combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()
