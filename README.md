# Multi-QR Code Recognition for Medicine Packs

This repository contains the code for detecting and decoding multiple QR codes on medicine packs.

---

## Project Structure

```
multiqr-hackathon/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── train.py                         # Training script (YOLOv8)
├── eval.py                          # Model Evaluation
├── visualizeInference.py            # Visualiztion model outputs
├── infer.py                         # Inference + QR decoding script
├── data/                            # Placeholder for dataset (not included)
│   └── demo_images/                 # Example images for testing
├── outputs/                         
│   ├── submission_detection_1.json  # Detection results on test folder
│   └── submission_decoding_2.json   # Decoding + classification JSON results on test folder
└── src/                              # Model, utils, dataset loaders, etc.
    ├── models/
    ├── datasets/
    ├── utils/
    └── __init__.py
```

---

## Environment Setup

1. Python 3.10+ is recommended.
2. Install dependencies:

```pip install -r requirements.txt```

**Dependencies include**:
- ultralytics (YOLOv8)
- torch
- opencv-python
- numpy
- pyzbar
- Pillow

---

## Training Instructions

The model was trained on **Google Colab**.  

**Steps to train:**

1. Open a Google Colab notebook.
2. Run the ```train.ipynb``` file

3. Upload your dataset (Roboflow or local) to Colab or use the given dataset in ```train.ipynb```
4. Train the model by running:

5. After training, the **best weights** are saved at: ```runs/detect/exp/weights/best.pt```

6. Download ```best.pt``` and save it under ```src/models```


---

## Inference Instructions

Run inference on a folder of images using:

```python infer.py```

Creates both detection_on_demo_1.json and decoding_on_demo_2.json files.

**Arguments which can be included with ```python indef.py```**:

| Argument           | Description                                           | Default |
|-------------------|-------------------------------------------------------|---------|
| --input          | Folder containing input images                        | data/demo_images |
| --output         | Path to save detection JSON (bbox only)              | outputs/detection_on_demo_1.json |
| --decoding_output| Path to save decoded QR JSON (bbox + value)          | outputs/decoding_on_demo_2.json |
| --weights        | YOLOv8 weights file path                              | src/models/best.pt |
| --conf           | Detection confidence threshold                         | 0.69    |
| --iou            | IoU threshold for non-max suppression                 | 0.45    |
| --device         | Device for inference (cpu or cuda)                   | cpu   |

**Outputs**:

1. submission_detection_1.json – detection results only:
```
[
  {
    "image_id": "img001",
    "qrs": [
      {"bbox": [x_min, y_min, x_max, y_max]},
      {"bbox": [x_min, y_min, x_max, y_max]}
    ]
  }
]
```

2. submission_decoding_2.json – detection + decoded QR values:
```
[
  {
    "image_id": "img001",
    "qrs": [
      {"bbox": [x_min, y_min, x_max, y_max], "value": "B12345"},
      {"bbox": [x_min, y_min, x_max, y_max], "value": "MFR56789"}
    ]
  }
]
```

---

## Notes

- The detection model is YOLOv8 trained on my Roboflow-annotated dataset.
- QR decoding uses pyzbar with high-contrast preprocessing and multiple rotations..
- Both JSON outputs are automatically created when running infer.py.

---
