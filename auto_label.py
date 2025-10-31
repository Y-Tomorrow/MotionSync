#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ultralytics import YOLO
import os

def save_yolo_pose_txt_with_bbox(results, save_dir, class_id=0):
    """
    保存每张图片的txt标注，每行包含bbox + 关键点
    """
    os.makedirs(save_dir, exist_ok=True)

    for r in results:
        path = r.path
        base = os.path.splitext(os.path.basename(path))[0]
        txt_path = os.path.join(save_dir, f"{base}.txt")

        with open(txt_path, "w") as f:
            for box, keypoints in zip(r.boxes.xywhn, r.keypoints.xyn):
                line = [class_id]  # 类别ID
                line.extend(box.tolist())  # bbox [x_c, y_c, w, h]，归一化
                for (x, y) in keypoints:
                    line.extend([float(x.item()), float(y.item()), 2])  # 关键点
                f.write(" ".join(map(str, line)) + "\n")

        print(f"✅ Saved {txt_path}")


def auto_label_yolo_format(img_dir, model_path="yolov8n-pose.pt", save_dir="./datasets/labels/train", conf_threshold=0.5):
    model = YOLO(model_path)
    imgs = [os.path.join(img_dir, x) for x in os.listdir(img_dir) if x.lower().endswith(('.jpg', '.png'))]

    print(f"🔍 Processing {len(imgs)} images...")
    
    for img_path in imgs:
        print(f"\n📸 Processing: {os.path.basename(img_path)}")
        results = model(img_path, conf=conf_threshold)
        
        # 调试信息
        for i, r in enumerate(results):
            print(f"  Result {i}: {len(r.boxes)} detections, {len(r.keypoints)} keypoint sets")
            if len(r.boxes) > 0:
                print(f"    Confidence scores: {r.boxes.conf.tolist()}")
        
        save_yolo_pose_txt_with_bbox(results, save_dir)

if __name__ == "__main__":
    auto_label_yolo_format(
        "./datasets/images/train",
        model_path="./models/yolov8n-pose.pt",
        save_dir="./datasets/labels/train",
        conf_threshold=0.5  # 提高置信度阈值，减少误检
    )
