#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ultralytics import YOLO
import os

def save_yolo_pose_txt(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for r in results:
        path = r.path
        base = os.path.splitext(os.path.basename(path))[0]
        txt_path = os.path.join(save_dir, f"{base}.txt")

        with open(txt_path, "w") as f:
            for box, keypoints in zip(r.boxes.xywhn, r.keypoints.xyn):
                # box: [x_center, y_center, w, h] (normalized)
                # keypoints: Nx2 normalized keypoints
                line = [0]  # class id
                line.extend(box.tolist())
                for (x, y) in keypoints:
                    line.extend([float(x.item()), float(y.item()), 2])  # 2=visible
                f.write(" ".join(map(str, line)) + "\n")

        print(f"✅ Saved {txt_path}")


def auto_label_yolo_format(img_dir, model_path="yolov8n-pose.pt", save_dir="./datasets/labels/train"):
    model = YOLO(model_path)
    imgs = [os.path.join(img_dir, x) for x in os.listdir(img_dir) if x.lower().endswith(('.jpg', '.png'))]

    for img in imgs:
        results = model(img)
        save_yolo_pose_txt(results, save_dir)


if __name__ == "__main__":
    auto_label_yolo_format(
        "./datasets/images/train",
        model_path="yolov8n-pose.pt",
        save_dir="./datasets/labels/train"
    )
