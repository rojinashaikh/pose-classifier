# main.py
import cv2
import mediapipe as mp
from ultralytics import YOLO
import os
import math

# Load input and output video paths
input_video = "C:\Users\Rojina.Shaikh\Downloads\MicrosoftTeams-video.mp4"
output_video = "yolo.mp4"
USE_MEDIAPIPE = True
USE_YOLO = True

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) if USE_MEDIAPIPE else None

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt") if USE_YOLO else None

# Open video file
cap = cv2.VideoCapture(input_video)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

out = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"\U0001F4FA Processing {frame_count} frames from {input_video}...")

def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return abs(ang + 360) if ang < 0 else abs(ang)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 detection
    if yolo_model:
        yolo_results = yolo_model(frame, verbose=False)
        frame = yolo_results[0].plot()

    # MediaPipe Pose detection
    if pose:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            landmark_spec = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
            connection_spec = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)

            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_spec,
                connection_spec
            )

            # Extra connections
            def draw_extra_connection(p1, p2):
                x1 = int(landmarks[p1].x * frame_width)
                y1 = int(landmarks[p1].y * frame_height)
                x2 = int(landmarks[p2].x * frame_width)
                y2 = int(landmarks[p2].y * frame_height)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            extra_connections = [
                (0, 1), (0, 4), (9, 10), (11, 13), (12, 14),
                (13, 15), (14, 16), (23, 24), (11, 23), (12, 24),
                (23, 25), (24, 26), (25, 27), (26, 28), (27, 31), (28, 32)
            ]
            for connection in extra_connections:
                draw_extra_connection(*connection)

            # Pose classification
            pose_label = "Unknown"
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_knee = landmarks[25]
            right_knee = landmarks[26]

            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_y = (left_hip.y + right_hip.y) / 2
            knee_y = (left_knee.y + right_knee.y) / 2

            vertical_diff = abs(shoulder_y - hip_y)
            hip_knee_diff = abs(hip_y - knee_y)

            if vertical_diff < 0.1:
                pose_label = "Lying Down"
            elif hip_knee_diff < 0.1:
                pose_label = "Standing"
            else:
                pose_label = "Sitting"

            cv2.putText(frame, f'Pose: {pose_label}', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
if pose:
    pose.close()
print(f"\u2705 Saved processed video to: {output_video}")

