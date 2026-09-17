# Development of Indigenous Contactless Integrated Track Monitoring Systems (ITMS) for Track Recording

A software-based railway track inspection prototype using computer vision and deep learning for automated railway fastener detection and condition classification.

## Project Overview

This project demonstrates an AI-assisted visual inspection pipeline inspired by the visual inspection component of Integrated Track Monitoring Systems (ITMS).

The system uses the Railway Fastener Defect Dataset (RFDD) and a YOLO26n object-detection model to detect and classify railway fasteners into six condition classes.

The detected results are converted into structured inspection records and presented through annotated images, CSV files, an HTML inspection report, a dashboard, and a simulated inspection video.

> **Note:** This is an academic software prototype. It is not a complete industrial railway ITMS and should not be used for engineering or maintenance decisions without appropriate validation.

## System Pipeline

Railway Images / Video
↓
Frame Extraction & Preprocessing
↓
YOLO26n Object Detection
↓
Fastener Detection & Classification
↓
Normal / Defective Classification
↓
Automated Inspection Recording
↓
CSV + Annotated Images + Video
↓
HTML Dashboard / Inspection Report

## Dataset

### Railway Fastener Defect Dataset (RFDD)

The project uses the RFDD dataset.

Dataset characteristics:

- 1,350 high-resolution full-scene images
- More than 8,100 fastener instances
- Image resolution: 2048 × 2021
- 6 fastener-condition classes
- Official train/validation/test split preserved

### Classes

| ID | Class |
|---|---|
| 0 | Deformed |
| 1 | Fractured |
| 2 | Missing |
| 3 | Inverted |
| 4 | Normal |
| 5 | Displaced |

## Model

**YOLO26n**

Training:

- Epochs: 40
- Dataset: RFDD
- Task: Object Detection
- Final model: `model/best.pt`

## Final Test Performance

Evaluation was performed on the official RFDD test set containing 100 images.

| Metric | Result |
|---|---:|
| Precision | 94.1% |
| Recall | 88.8% |
| mAP@50 | 94.5% |
| mAP@50–95 | 70.7% |

### Class-wise mAP@50

| Class | mAP@50 |
|---|---:|
| Deformed | 80.7% |
| Fractured | 97.3% |
| Missing | 93.8% |
| Inverted | 99.5% |
| Normal | 99.2% |
| Displaced | 96.4% |

## Repository Structure

```text
ITMS-Github/
│
├── config/
│   └── data.yaml
│
├── demo/
│   ├── ITMS_inspection_browser.mp4
│   └── simulated_track_inspection.mp4
│
├── docs/
│   ├── ITMS_Final_Project_Report_Complete.pdf
│   ├── ITMS_Project_Presentation.pptx
│   └── ITMS_Viva_and_Project_Defense_Preparation.pdf
│
├── model/
│   └── best.pt
│
├── outputs/
│   ├── ITMS_Dashboard.html
│   ├── ITMS_Inspection_Report.html
│   ├── track_inspection_record.csv
│   └── video_inspection_record.csv
│
├── results/
│   ├── BoxF1_curve.png
│   ├── BoxPR_curve.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   └── results.png
│
├── src/
│   ├── create_test_video.py
│   ├── itms_track_recording.py
│   └── itms_video_monitor.py
│
├── .gitignore
├── README.md
└── requirements.txt
