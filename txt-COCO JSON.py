#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_pose_coco_fixed.py
双向转换：
  YOLOv8-Pose TXT <-> COCO JSON
特点：
- 转回 TXT 时保持原 bbox 不变
- 支持 17 个关键点，可见性 0/1/2
"""

import os, json, argparse

NUM_KEYPOINTS = 17

# -----------------------------
# TXT -> COCO JSON
# -----------------------------
def txt_to_coco(img_dir, txt_dir, save_path):
    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    for txt_file in sorted(os.listdir(txt_dir)):
        if not txt_file.endswith(".txt"):
            continue
        base = os.path.splitext(txt_file)[0]
        # 支持 jpg/png
        img_path = None
        for ext in [".jpg", ".png"]:
            path = os.path.join(img_dir, base + ext)
            if os.path.exists(path):
                img_path = path
                break
        if img_path is None:
            print(f"⚠️ Skip {base}: image not found")
            continue

        from PIL import Image
        img = Image.open(img_path)
        w, h = img.size

        images.append({
            "id": img_id,
            "file_name": os.path.basename(img_path),
            "width": w,
            "height": h
        })

        # 读取 txt
        with open(os.path.join(txt_dir, txt_file)) as f:
            lines = f.readlines()

        for line in lines:
            arr = line.strip().split()
            if len(arr) < 5 + NUM_KEYPOINTS * 3:
                print(f"⚠️ Skip {txt_file}: invalid format")
                continue

            # 保留原始 bbox
            orig_box = [float(x) for x in arr[1:6]]
            keypoints_txt = arr[5:]

            keypoints = []
            for i in range(NUM_KEYPOINTS):
                x = float(keypoints_txt[i*3]) * w
                y = float(keypoints_txt[i*3+1]) * h
                v = int(keypoints_txt[i*3+2])
                keypoints.extend([x, y, v])

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": orig_box,  # 保存原 bbox
                "keypoints": keypoints,
                "num_keypoints": NUM_KEYPOINTS
            })
            ann_id += 1

        img_id += 1

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person",
                        "keypoints": [f"kp{i}" for i in range(NUM_KEYPOINTS)],
                        "skeleton": []}]
    }

    with open(save_path, "w") as f:
        json.dump(coco_data, f, indent=2)
    print(f"✅ Saved COCO JSON: {save_path}")


# -----------------------------
# COCO JSON -> TXT
# -----------------------------
def coco_to_txt(coco_json_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(coco_json_path) as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}

    for ann in data['annotations']:
        img = images[ann['image_id']]
        w, h = img['width'], img['height']

        txt_path = os.path.join(save_dir, img['file_name'].replace(".jpg", ".txt").replace(".png", ".txt"))

        # 使用原始 bbox
        orig_box = ann.get('bbox', [0, 0.5, 0.5, 1.0, 1.0])
        line = [0] + orig_box

        # keypoints
        for i in range(ann['num_keypoints']):
            x = ann['keypoints'][i*3] / w
            y = ann['keypoints'][i*3+1] / h
            v = ann['keypoints'][i*3+2]
            line.extend([x, y, v])

        with open(txt_path, 'w') as f:
            f.write(" ".join(map(str, line)) + "\n")
        print(f"✅ Saved TXT: {txt_path}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8-Pose TXT <-> COCO JSON (保持原 bbox)")
    parser.add_argument("--mode", choices=["txt2coco", "coco2txt"], required=True, help="转换方向")
    parser.add_argument("--img_dir", help="图片目录")
    parser.add_argument("--txt_dir", help="TXT 标签目录")
    parser.add_argument("--json_path", help="COCO JSON 文件路径")
    parser.add_argument("--save_dir", help="TXT 输出目录或 COCO JSON 保存路径")
    args = parser.parse_args()

    if args.mode == "txt2coco":
        assert args.img_dir and args.txt_dir and args.save_dir
        txt_to_coco(args.img_dir, args.txt_dir, args.save_dir)
    else:
        assert args.json_path and args.save_dir
        coco_to_txt(args.json_path, args.save_dir)
