from ultralytics import YOLO
import os
import json
import cv2
pwd = os.getcwd()

MODEL_PATH = os.path.join(pwd, "src/models/best.pt")
TEST_FOLDER = os.path.join(pwd,"src/dataset/test/images")
TEST_LABEL_FOLDER = os.path.join(pwd,"src/dataset/test/labels") 
PRED_JSON = os.path.join(pwd,"outputs_test/test_detection_1.json")
IOU_THR = 0.5

os.makedirs("outputs_test", exist_ok=True)

model = YOLO(MODEL_PATH)
results = model.predict(source=TEST_FOLDER, conf=0.69)


submission = []
for img_path, result in zip(sorted(os.listdir(TEST_FOLDER)), results):
    img_id = os.path.splitext(img_path)[0]
    qrs = []
    for box in result.boxes.xyxy.cpu().numpy():
        qrs.append({"bbox": box.tolist()})
    submission.append({"image_id": img_id, "qrs": qrs})

with open(PRED_JSON, "w") as f:
    json.dump(submission, f, indent=2)
print(f"Predictions saved to {PRED_JSON}")


def load_gt_from_txt(test_folder, label_folder):
    gt = []
    for img_file in sorted(os.listdir(test_folder)):
        img_id = os.path.splitext(img_file)[0]
        img_path = os.path.join(test_folder, img_file)
        h, w = cv2.imread(img_path).shape[:2]

        txt_file = os.path.join(label_folder, img_id + ".txt")
        qrs = []
        if os.path.exists(txt_file):
            with open(txt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id, x_c, y_c, bw, bh = map(float, parts)
                    x_min = (x_c - bw / 2) * w
                    y_min = (y_c - bh / 2) * h
                    x_max = (x_c + bw / 2) * w
                    y_max = (y_c + bh / 2) * h
                    qrs.append({"bbox": [x_min, y_min, x_max, y_max]})
        gt.append({"image_id": img_id, "qrs": qrs})
    return gt

GT = load_gt_from_txt(TEST_FOLDER, TEST_LABEL_FOLDER)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def evaluate(pred_json, gt_list, iou_thr=0.5):
    preds = json.load(open(pred_json))
    gt_map = {item["image_id"]: item["qrs"] for item in gt_list}

    total_gt, total_matched, total_pred = 0, 0, 0
    for p in preds:
        image_id = p["image_id"]
        pred_boxes = [q["bbox"] for q in p["qrs"]]
        gt_boxes = [q["bbox"] for q in gt_map.get(image_id, [])]
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        used = set()
        for pb in pred_boxes:
            best_iou, best_idx = 0, -1
            for j, gb in enumerate(gt_boxes):
                if j in used: continue
                cur_iou = iou(pb, gb)
                if cur_iou > best_iou:
                    best_iou, best_idx = cur_iou, j
            if best_iou >= iou_thr:
                total_matched += 1
                used.add(best_idx)

    precision = total_matched / total_pred if total_pred > 0 else 0
    recall = total_matched / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n results")
    print(f"gt boxes: {total_gt}, predicted: {total_pred}, matched: {total_matched}")
    print(f"precision: {precision:.4f}, recall: {recall:.4f}, f1_score: {f1:.4f}")

# === 4️⃣ Run evaluation ===
evaluate(PRED_JSON, GT, IOU_THR)
