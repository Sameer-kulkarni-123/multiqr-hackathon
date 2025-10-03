# infer.py
from ultralytics import YOLO
from pathlib import Path
import json
import numpy as np
import os
import cv2
from pyzbar.pyzbar import decode
from PIL import Image
import argparse

# === Helper Functions ===
def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 inference + QR decoding")
    parser.add_argument('--input', type=str, default="data/demo_images", help='Folder with test images')
    parser.add_argument('--output', type=str, default="outputs/detection_on_demo_1.json", help='Detection JSON output path')
    parser.add_argument('--decoding_output', type=str, default="outputs/decoding_on_demo_2.json", help='Decoded QR JSON output path')
    parser.add_argument('--weights', type=str, default="src/models/best.pt", help='Path to YOLOv8 weights')
    parser.add_argument('--conf', type=float, default=0.69, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--device', type=str, default="cpu", help='"cuda" or "cpu"')
    return parser.parse_args()

def image_id_from_path(path: Path):
    return path.stem  # e.g. "img001"

def xyxy_to_list(box):
    x1, y1, x2, y2 = box
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

def decode_qr_from_crop(crop):
    """Decode a cropped QR code with high-contrast + rotation."""
    if crop.size == 0:
        return ""
    
    # Resize small crops to improve decoding
    crop = cv2.resize(crop, (300, 300), interpolation=cv2.INTER_LINEAR)
    
    # Grayscale + threshold for high contrast
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Try multiple rotations
    rotation_map = {90: cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE}

    for angle in [0, 90, 180, 270]:
        rotated = thresh if angle == 0 else cv2.rotate(thresh, rotation_map[angle])
        decoded_objs = decode(Image.fromarray(rotated))
        if decoded_objs:
            return decoded_objs[0].data.decode("utf-8")
    
    return ""  # return empty if not decoded

# === Main Function ===
def main():
    args = parse_args()
    model = YOLO(args.weights)
    input_folder = Path(args.input)
    image_paths = sorted([p for p in input_folder.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])

    detection_results = []
    decoding_results = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        preds = model.predict(source=str(img_path), conf=args.conf, iou=args.iou, device=args.device, verbose=False)
        r = preds[0]

        qrs_det = []
        qrs_dec = []

        boxes = r.boxes.xyxy.numpy() if hasattr(r.boxes, 'xyxy') else np.array([])

        for box in boxes:
            bbox_list = xyxy_to_list(box)
            qrs_det.append({"bbox": bbox_list})

            x1, y1, x2, y2 = bbox_list
            crop = img[y1:y2, x1:x2]
            qr_value = decode_qr_from_crop(crop)
            qrs_dec.append({"bbox": bbox_list, "value": qr_value})

        image_id = image_id_from_path(img_path)
        detection_results.append({"image_id": image_id, "qrs": qrs_det})
        decoding_results.append({"image_id": image_id, "qrs": qrs_dec})

    # Save detection JSON
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(detection_results, f, indent=2)
    print(f"✅ Detection JSON saved to {args.output}")

    # Save decoding JSON
    Path(args.decoding_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.decoding_output, "w") as f:
        json.dump(decoding_results, f, indent=2)
    print(f"✅ Decoding JSON saved to {args.decoding_output}")

if __name__ == "__main__":
    main()
