from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
pwd = os.getcwd()

MODEL = os.path.join(pwd, "rsrc/models/best.pt")   # trained yolov8 weights
INPUT_FOLDER = os.path.join(pwd, "data/demo_images") 

model = YOLO(MODEL)  


img_folder = INPUT_FOLDER 

results = model.predict(source=img_folder, conf=0.69)  # device=0 for GPU

for i, result in enumerate(results):
    img_with_boxes = result.plot()  # draw bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(img_with_boxes)
    plt.axis("off")
    plt.title(f"Prediction {i}")
    plt.show()
