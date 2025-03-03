import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path).to("cuda")
        with open("coco.txt", "r") as f:
            self.class_list = f.read().strip().split("\n")
        if "person" not in self.class_list:
            raise ValueError("Error: 'person' class not found in class_list. Check coco.txt file!")

    def detect(self, frame):
        results = self.model.predict(frame)
        detections = results[0].boxes.data.cpu().numpy()
        px = pd.DataFrame(detections).astype("float")

        people_list = []
        for _, row in px.iterrows():
            x1, y1, x2, y2 = map(int, row[:4])
            class_id = int(row[5])
            if 0 <= class_id < len(self.class_list) and self.class_list[class_id] == "person":
                people_list.append([x1, y1, x2, y2])

        return people_list
