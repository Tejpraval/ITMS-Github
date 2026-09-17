import cv2
import os
import csv
from ultralytics import YOLO

# -------------------------------------------------
# Paths
# -------------------------------------------------

MODEL_PATH = r"model/best.pt"

VIDEO_PATH = r"demo/simulated_track_inspection.mp4"
OUTPUT_DIR = r"outputs/video_inspection"

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "ITMS_inspection_result.mp4"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "video_inspection_record.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Classes
# -------------------------------------------------

CLASS_NAMES = [
    "Deformed",
    "Fractured",
    "Missing",
    "Inverted",
    "Normal",
    "Displaced"
]

# -------------------------------------------------
# Load model
# -------------------------------------------------

print("Loading ITMS YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")

# -------------------------------------------------
# Open video
# -------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print()
print("Video information:")
print("FPS:", fps)
print("Resolution:", width, "x", height)
print("Total frames:", total_frames)

# -------------------------------------------------
# Video writer
# -------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

# -------------------------------------------------
# CSV
# -------------------------------------------------

csv_file = open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "Frame",
    "Timestamp_seconds",
    "Condition",
    "Confidence",
    "X1",
    "Y1",
    "X2",
    "Y2"
])

# -------------------------------------------------
# Process video
# -------------------------------------------------

frame_number = 0
total_detections = 0
total_defects = 0

print()
print("Starting ITMS video inspection...")
print()

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # Run YOLO
    results = model(
        frame,
        imgsz=640,
        device="cpu",
        verbose=False
    )

    result = results[0]

    # Draw detections
    annotated_frame = result.plot()

    # Process detections
    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            condition = CLASS_NAMES[class_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            timestamp = frame_number / fps

            csv_writer.writerow([
                frame_number,
                round(timestamp, 2),
                condition,
                round(confidence, 3),
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ])

            total_detections += 1

            if condition != "Normal":
                total_defects += 1

    # Add ITMS information
    cv2.putText(
        annotated_frame,
        f"ITMS Frame: {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Defects detected: {total_defects}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    writer.write(annotated_frame)

    if frame_number % 10 == 0:

        print(
            f"Processed {frame_number}/{total_frames} frames"
        )

# -------------------------------------------------
# Finish
# -------------------------------------------------

cap.release()
writer.release()
csv_file.close()

print()
print("========================================")
print(" ITMS VIDEO INSPECTION COMPLETED")
print("========================================")
print()
print("Frames processed :", frame_number)
print("Fastener detections :", total_detections)
print("Defect detections :", total_defects)
print()
print("Annotated video:")
print(OUTPUT_VIDEO)
print()
print("Inspection CSV:")
print(OUTPUT_CSV)
