from ultralytics import YOLO
import os
import csv

# -------------------------------------------------
# Paths
# -------------------------------------------------

MODEL_PATH = r"model/best.pt"
INPUT_FOLDER = r"images/test"
OUTPUT_FOLDER = r"outputs"
ANNOTATED_FOLDER = os.path.join(OUTPUT_FOLDER, "annotated_images")

CSV_FILE = os.path.join(OUTPUT_FOLDER, "track_inspection_record.csv")

# -------------------------------------------------
# Create output folders
# -------------------------------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# -------------------------------------------------
# Load trained model
# -------------------------------------------------

print("Loading ITMS detection model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print()

# -------------------------------------------------
# Class names
# -------------------------------------------------

class_names = {
    0: "Deformed",
    1: "Fractured",
    2: "Missing",
    3: "Inverted",
    4: "Normal",
    5: "Displaced"
}

# -------------------------------------------------
# CSV file
# -------------------------------------------------

csv_file = open(CSV_FILE, "w", newline="", encoding="utf-8-sig")

writer = csv.writer(csv_file)

writer.writerow([
    "Image",
    "Fastener_ID",
    "Condition",
    "Confidence",
    "X1",
    "Y1",
    "X2",
    "Y2",
    "Status"
])

# -------------------------------------------------
# Process images
# -------------------------------------------------

image_files = []

for filename in os.listdir(INPUT_FOLDER):

    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_files.append(filename)

image_files.sort()

print("Images found:", len(image_files))
print()

total_defects = 0
total_fasteners = 0

# -------------------------------------------------
# Run detection
# -------------------------------------------------

for image_number, filename in enumerate(image_files, start=1):

    image_path = os.path.join(INPUT_FOLDER, filename)

    print(
        f"[{image_number}/{len(image_files)}] "
        f"Inspecting {filename}"
    )

    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.25,
        device="cpu",
        verbose=False
    )

    result = results[0]

    boxes = result.boxes

    fastener_id = 0
    image_has_defect = False

    if boxes is not None:

        for box in boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            condition = class_names[class_id]

            coordinates = box.xyxy[0].tolist()

            x1 = round(coordinates[0], 2)
            y1 = round(coordinates[1], 2)
            x2 = round(coordinates[2], 2)
            y2 = round(coordinates[3], 2)

            fastener_id += 1
            total_fasteners += 1

            if condition == "Normal":
                status = "NORMAL"
            else:
                status = "DEFECT"
                image_has_defect = True
                total_defects += 1

            writer.writerow([
                filename,
                fastener_id,
                condition,
                round(confidence, 4),
                x1,
                y1,
                x2,
                y2,
                status
            ])

    # -------------------------------------------------
    # Save annotated image
    # -------------------------------------------------

    annotated_image = result.plot()

    output_image = os.path.join(
        ANNOTATED_FOLDER,
        filename
    )

    from PIL import Image

    Image.fromarray(annotated_image[:, :, ::-1]).save(
        output_image
    )

    if image_has_defect:
        print("   STATUS: DEFECT DETECTED")
    else:
        print("   STATUS: NORMAL")

# -------------------------------------------------
# Finish
# -------------------------------------------------

csv_file.close()

print()
print("========================================")
print(" ITMS TRACK INSPECTION COMPLETED")
print("========================================")
print()
print("Total images inspected :", len(image_files))
print("Total fasteners detected:", total_fasteners)
print("Total defects detected :", total_defects)
print()
print("Inspection CSV:")
print(CSV_FILE)
print()
print("Annotated images:")
print(ANNOTATED_FOLDER)
