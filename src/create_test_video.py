import os
import cv2
import numpy as np
from PIL import Image

# -----------------------------------------
# Paths
# -----------------------------------------

INPUT_FOLDER = r"images/test"

OUTPUT_VIDEO = r"demo/simulated_track_inspection.mp4"

# -----------------------------------------
# Get images
# -----------------------------------------

image_files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

image_files.sort()

print("Images found:", len(image_files))

# -----------------------------------------
# Video settings
# -----------------------------------------

fps = 10

video_width = 1280
video_height = 720

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (video_width, video_height)
)

# -----------------------------------------
# Add images to video
# -----------------------------------------

for i, filename in enumerate(image_files, start=1):

    image_path = os.path.join(INPUT_FOLDER, filename)

    # PIL handles Unicode filenames correctly
    image = Image.open(image_path).convert("RGB")

    image = np.array(image)

    # RGB -> BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize to video resolution
    image = cv2.resize(
        image,
        (video_width, video_height)
    )

    # Add frame number
    cv2.putText(
        image,
        f"Inspection Frame: {i}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    video.write(image)

    if i % 10 == 0:
        print(f"Added {i}/{len(image_files)} frames")

# -----------------------------------------
# Finish
# -----------------------------------------

video.release()

print()
print("========================================")
print(" SIMULATED ITMS VIDEO CREATED")
print("========================================")
print()
print("Video:")
print(OUTPUT_VIDEO)
