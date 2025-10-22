#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_pose_labels.py
双向转换：
  YOLO Pose <-> LabelMe JSON
"""

import os, json, cv2, base64, argparse

def yolo_pose_txt_to_labelme(img_dir, label_dir, save_dir, num_keypoints=17):
    os.makedirs(save_dir, exist_ok=True)
    for txt_file in os.listdir(label_dir):
        if not txt_file.endswith(".txt"):
            continue
        base = os.path.splitext(txt_file)[0]

        # 支持jpg/png
        img_path = None
        for ext in [".jpg", ".png"]:
            path = os.path.join(img_dir, base + ext)
            if os.path.exists(path):
                img_path = path
                break
        if img_path is None:
            print(f"⚠️ Skip {base}: no matching image found")
            continue

        label_path = os.path.join(label_dir, txt_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Skip {img_path}: cannot read image")
            continue

        h, w = image.shape[:2]

        with open(img_path, "rb") as f:
            imageData = base64.b64encode(f.read()).decode("utf-8")

        with open(label_path, "r") as f:
            data = f.readlines()

        shapes = []
        for line in data:
            arr = line.strip().split()
            if len(arr) < 5 + num_keypoints * 3:
                print(f"⚠️ Skip {txt_file}: unexpected data length")
                continue
            keypoints = arr[5:]
            for i in range(num_keypoints):
                x = float(keypoints[i*3]) * w
                y = float(keypoints[i*3+1]) * h
                v = int(float(keypoints[i*3+2]))
                shapes.append({
                    "label": f"keypoint_{i}",
                    "points": [[x, y]],
                    "shape_type": "point",
                    "flags": {"visible": v}
                })

        labelme_data = {
            "version": "5.3.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(img_path),
            "imageData": imageData,
            "imageHeight": h,
            "imageWidth": w
        }

        out_path = os.path.join(save_dir, base + ".json")
        with open(out_path, "w") as f:
            json.dump(labelme_data, f, indent=2)
        print(f"✅ Saved JSON: {out_path}")


def labelme_to_yolo_pose(labelme_dir, save_dir, img_dir, num_keypoints=17):
    os.makedirs(save_dir, exist_ok=True)
    for js in os.listdir(labelme_dir):
        if not js.endswith(".json"):
            continue

        json_path = os.path.join(labelme_dir, js)
        with open(json_path, "r") as f:
            data = json.load(f)

        img_path = os.path.join(img_dir, data["imagePath"])
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            continue

        h, w = data["imageHeight"], data["imageWidth"]
        keypoints = sorted(
            [kp for kp in data["shapes"] if kp["shape_type"] == "point"],
            key=lambda x: int(x["label"].split("_")[1])
        )

        # dummy box, 因为LabelMe里没框
        line = [0, 0.5, 0.5, 1.0, 1.0]

        for kp in keypoints[:num_keypoints]:
            x = kp["points"][0][0] / w
            y = kp["points"][0][1] / h
            v = int(kp["flags"].get("visible", 2))
            line.extend([x, y, v])

        out_path = os.path.join(save_dir, js.replace(".json", ".txt"))
        with open(out_path, "w") as f:
            f.write(" ".join(map(str, line)) + "\n")
        print(f"✅ Saved TXT: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO Pose <-> LabelMe JSON")
    parser.add_argument("--mode", choices=["txt2json", "json2txt"], required=True, help="转换方向")
    parser.add_argument("--img_dir", required=True, help="图片文件夹路径")
    parser.add_argument("--label_dir", required=True, help="输入标签文件夹路径")
    parser.add_argument("--save_dir", required=True, help="输出文件夹路径")
    parser.add_argument("--num_keypoints", type=int, default=17, help="关键点数量(默认17)")
    args = parser.parse_args()

    if args.mode == "txt2json":
        yolo_pose_txt_to_labelme(args.img_dir, args.label_dir, args.save_dir, args.num_keypoints)
    else:
        labelme_to_yolo_pose(args.label_dir, args.save_dir, args.img_dir, args.num_keypoints)
