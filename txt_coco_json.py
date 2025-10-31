#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_pose_coco_fixed.py
双向转换：
  YOLOv8-Pose TXT <-> COCO JSON
特点：
- 转回 TXT 时保持原 bbox 不变
- 支持 17 个关键点，可见性 0/1/2
- 自动补全缺失关键点和 bbox
"""

import os, json, argparse

NUM_KEYPOINTS = 17

# -----------------------------
# TXT -> COCO JSON
# -----------------------------
def txt_to_coco(img_dir, txt_dir, save_path):
    from PIL import Image
    images, annotations = [], []
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

        img = Image.open(img_path)
        w, h = img.size
        
        # COCO Annotator格式的图片信息
        images.append({
            "id": img_id,
            "dataset_id": 1,
            "category_ids": [],
            "path": f"/datasets/images/train/{os.path.basename(img_path)}",
            "width": w,
            "height": h,
            "file_name": os.path.basename(img_path),
            "annotated": False,
            "annotating": [],
            "num_annotations": 0,
            "metadata": {},
            "deleted": False,
            "milliseconds": 0,
            "events": [],
            "regenerate_thumbnail": False
        })

        with open(os.path.join(txt_dir, txt_file)) as f:
            lines = f.readlines()

        for line in lines:
            arr = line.strip().split()
            if len(arr) < 1 + 4 + NUM_KEYPOINTS * 3:
                arr += ['0'] * (1 + 4 + NUM_KEYPOINTS * 3 - len(arr))

            # 转换 bbox
            x_c, y_c, w_norm, h_norm = [float(x) for x in arr[1:5]]
            bbox_w = w_norm * w
            bbox_h = h_norm * h
            x_min = x_c * w - bbox_w / 2
            y_min = y_c * h - bbox_h / 2
            orig_box = [x_min, y_min, bbox_w, bbox_h]

            # keypoints
            keypoints_txt = arr[5:]
            keypoints = []
            for i in range(NUM_KEYPOINTS):
                x = float(keypoints_txt[i * 3]) * w
                y = float(keypoints_txt[i * 3 + 1]) * h
                v = int(float(keypoints_txt[i * 3 + 2]))
                keypoints.extend([x, y, v])

            # 创建segmentation多边形 (bbox的四个角点)
            x, y, w_box, h_box = orig_box
            segmentation = [
                [x + w_box, y, x + w_box, y + h_box, x, y + h_box, x, y]
            ]

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": segmentation,
                "area": int(w_box * h_box),
                "bbox": orig_box,
                "iscrowd": False,
                "isbbox": True,
                "color": "#e02f84",
                "keypoints": keypoints,
                "metadata": {},
                "num_keypoints": NUM_KEYPOINTS
            })
            ann_id += 1

        img_id += 1

    # 更新images的num_annotations
    img_ann_count = {}
    for ann in annotations:
        img_id = ann['image_id']
        img_ann_count[img_id] = img_ann_count.get(img_id, 0) + 1
    
    for img in images:
        img['num_annotations'] = img_ann_count.get(img['id'], 0)
        img['annotated'] = img['num_annotations'] > 0

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "person",
            "supercategory": "",
            "color": "#3ccf69",
            "metadata": {},
            "keypoint_colors": [
                "#bf5c4d", "#d99100", "#4d8068", "#0d2b80", "#9c73bf",
                "#ff1a38", "#bf3300", "#736322", "#33fff1", "#3369ff",
                "#9d13bf", "#733941", "#ffb499", "#d0d957", "#0b5e73",
                "#0000ff", "#730b5e"
            ],
            "keypoints": [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ],
            "skeleton": [
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                [2, 4], [3, 5], [4, 6], [5, 7]
            ]
        }]
    }

    with open(save_path, "w") as f:
        json.dump(coco_data, f, indent=2)
    print(f"✅ Saved COCO Annotator JSON: {save_path}")


# -----------------------------
# COCO JSON -> TXT
# -----------------------------
def coco_to_txt(coco_json_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(coco_json_path) as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    
    # 按图片分组标注
    img_annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    # 为每张图片生成一个txt文件
    for img_id, annotations in img_annotations.items():
        img = images[img_id]
        w, h = img['width'], img['height']
        
        txt_path = os.path.join(save_dir, img['file_name'].replace(".jpg", ".txt").replace(".png", ".txt"))
        
        with open(txt_path, 'w') as f:
            for ann in annotations:
                # 转换bbox为YOLO格式 (归一化的中心点坐标和宽高)
                bbox = ann.get('bbox', [0, 0, 0, 0])
                if len(bbox) < 4:
                    bbox += [0] * (4 - len(bbox))
                
                x_min, y_min, w_box, h_box = bbox
                
                # 转换为YOLO格式: 中心点坐标和宽高，归一化
                x_center = (x_min + w_box / 2) / w
                y_center = (y_min + h_box / 2) / h
                w_norm = w_box / w
                h_norm = h_box / h
                
                line = [0, x_center, y_center, w_norm, h_norm]

                # keypoints，归一化
                kpts = ann.get('keypoints', [])
                for i in range(NUM_KEYPOINTS):
                    if i*3+2 < len(kpts):
                        x = kpts[i*3] / w
                        y = kpts[i*3+1] / h
                        v = int(kpts[i*3+2])
                    else:
                        x = y = 0.0
                        v = 0
                    line.extend([x, y, v])

                # 补齐总列数 56
                while len(line) < 1 + 4 + NUM_KEYPOINTS*3:
                    line.extend([0.0, 0.0, 0])

                f.write(" ".join(map(str, line)) + "\n")
        
        print(f"✅ Saved TXT: {txt_path} with {len(annotations)} annotations")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8-Pose TXT <-> COCO JSON (保持原 bbox)")
    parser.add_argument("--mode", choices=["txt2coco", "coco2txt"], required=True)
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
