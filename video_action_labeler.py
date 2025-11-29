#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频动作标注工具
从视频中提取pose关键点序列，并标注动作（W/A/S/D/空格跳跃）
"""

import os
import cv2
import json
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

# 动作类别映射
ACTION_MAP = {
    'w': 0,  # 前进
    'a': 1,  # 左
    's': 2,  # 后退
    'd': 3,  # 右
    ' ': 4,  # 跳跃（空格）
    'idle': 5  # 静止
}

ACTION_NAMES = {v: k for k, v in ACTION_MAP.items()}


def select_best_keypoints(result):
    """从单帧结果中选择一组关键点（优先置信度最高的目标）"""
    if not hasattr(result, 'keypoints') or result.keypoints is None or len(result.keypoints) == 0:
        return None
    kps = result.keypoints.xyn  # 归一化关键点 (num, 17, 2)
    if kps is None or len(kps) == 0:
        return None

    # 按置信度排序选第一
    best_idx = 0
    try:
        if hasattr(result, 'boxes') and result.boxes is not None and hasattr(result.boxes, 'conf'):
            confs = result.boxes.conf.cpu().numpy().reshape(-1)
            if len(confs) == len(kps):
                best_idx = int(np.argmax(confs))
    except Exception:
        best_idx = 0

    arr = kps[best_idx].cpu().numpy() if hasattr(kps, 'cpu') else kps[best_idx]
    if arr.shape[0] < 17:
        return None
    return arr[:17, :2]


def extract_keypoints_from_video(video_path, model_path, conf_threshold=0.5, roi=None):
    """从视频中提取所有帧的关键点序列"""
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频: {video_path}')
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: {total_frames}帧, {fps:.2f}fps")
    
    keypoint_sequence = []
    frame_indices = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h_full, w_full = frame.shape[:2]
        
        # 在ROI区域内推理
        if roi is not None:
            x, y, w_roi, h_roi = roi
            if x >= 0 and y >= 0 and x + w_roi <= w_full and y + h_roi <= h_full:
                roi_frame = frame[y:y+h_roi, x:x+w_roi]
                if roi_frame.size > 0:
                    results = model(roi_frame, conf=conf_threshold, verbose=False)
                else:
                    results = []
            else:
                results = []
        else:
            results = model(frame, conf=conf_threshold, verbose=False)
        
        # 提取关键点
        best_kp = None
        for r in results:
            kp_local = select_best_keypoints(r)
            if kp_local is not None:
                # 如果使用了ROI，转换坐标
                if roi is not None:
                    x, y, w_roi, h_roi = roi
                    kp_global_x = (kp_local[:, 0] * w_roi + x) / w_full
                    kp_global_y = (kp_local[:, 1] * h_roi + y) / h_full
                    best_kp = np.stack([kp_global_x, kp_global_y], axis=1)
                else:
                    best_kp = kp_local
                break
        
        if best_kp is not None:
            keypoint_sequence.append(best_kp.flatten().tolist())  # (34,) 展平
            frame_indices.append(frame_idx)
        else:
            # 如果该帧没有检测到关键点，用上一帧的关键点（如果存在）
            if len(keypoint_sequence) > 0:
                keypoint_sequence.append(keypoint_sequence[-1])
                frame_indices.append(frame_idx)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"已处理 {frame_idx}/{total_frames} 帧")
    
    cap.release()
    print(f"提取完成: {len(keypoint_sequence)} 帧关键点")
    return keypoint_sequence, frame_indices, fps


def label_video_interactive(video_path, model_path, output_path, conf_threshold=0.5, 
                           sequence_length=30, roi=None):
    """交互式标注视频动作"""
    print("=" * 60)
    print("视频动作标注工具")
    print("=" * 60)
    print("\n操作说明:")
    print("  W - 标注为前进")
    print("  A - 标注为左移")
    print("  S - 标注为后退")
    print("  D - 标注为右移")
    print("  空格 - 标注为跳跃")
    print("  I - 标注为静止")
    print("  ←/→ - 前进/后退10帧")
    print("  P - 播放/暂停")
    print("  Q - 退出并保存")
    print("  R - 重置当前片段标注")
    print("=" * 60)
    
    # 提取关键点
    print("\n正在提取关键点...")
    keypoint_sequence, frame_indices, fps = extract_keypoints_from_video(
        video_path, model_path, conf_threshold, roi
    )
    
    if len(keypoint_sequence) == 0:
        print("错误: 未能提取到关键点")
        return
    
    # 打开视频用于可视化
    cap = cv2.VideoCapture(video_path)
    total_frames = len(keypoint_sequence)
    
    # 标注数据: {start_frame: {end_frame: action_id}}
    annotations = {}
    current_frame = 0
    playing = False
    
    # 加载已有标注（如果存在）
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'annotations' in data:
                    annotations = {int(k): v for k, v in data['annotations'].items()}
                    print(f"已加载 {len(annotations)} 个标注片段")
        except Exception as e:
            print(f"加载已有标注失败: {e}")
    
    def draw_keypoints(frame, kp_2d, action_label=""):
        """在帧上绘制关键点"""
        h, w = frame.shape[:2]
        if kp_2d is None:
            return frame
        
        # 重塑为 (17, 2)
        kp = np.array(kp_2d).reshape(17, 2)
        pts_px = (kp * np.array([w, h])).astype(int)
        
        # 绘制关键点
        for (x, y) in pts_px:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
        
        # 绘制骨架连线
        edges = [
            [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6],
            [5, 7], [7, 9], [6, 8], [8, 10],
            [11, 13], [13, 15], [12, 14], [14, 16],
            [5, 6], [11, 12], [5, 11], [6, 12]
        ]
        for e in edges:
            a, b = e[0], e[1]
            if 0 <= a < len(pts_px) and 0 <= b < len(pts_px):
                ax, ay = pts_px[a]
                bx, by = pts_px[b]
                cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), (0, 200, 255), 2)
        
        # 显示动作标签
        if action_label:
            cv2.putText(frame, f"Action: {action_label}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def get_current_action():
        """获取当前帧所属的动作标注"""
        for start in sorted(annotations.keys(), reverse=True):
            if current_frame >= start:
                for end, action_id in annotations[start].items():
                    if current_frame <= end:
                        return ACTION_NAMES.get(action_id, "unknown")
        return "未标注"
    
    def save_annotations():
        """保存标注"""
        data = {
            'video_path': video_path,
            'total_frames': total_frames,
            'fps': fps,
            'keypoint_sequence': keypoint_sequence,
            'frame_indices': frame_indices,
            'annotations': {str(k): {str(ek): ev for ek, ev in v.items()} 
                          for k, v in annotations.items()},
            'action_map': ACTION_MAP
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n标注已保存到: {output_path}")
    
    print(f"\n开始标注 (总帧数: {total_frames})")
    print("按任意键开始...")
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
        
        # 获取当前帧的关键点
        kp_2d = None
        if current_frame < len(keypoint_sequence):
            kp_array = np.array(keypoint_sequence[current_frame]).reshape(17, 2)
            kp_2d = kp_array
        
        # 获取当前动作
        action_label = get_current_action()
        
        # 绘制
        frame = draw_keypoints(frame, kp_2d, action_label)
        
        # 显示信息
        info_text = [
            f"Frame: {current_frame}/{total_frames}",
            f"Action: {action_label}",
            f"Annotations: {len(annotations)} segments",
            "",
            "W/A/S/D/Space/I: Label | ←/→: Navigate | P: Play | Q: Quit"
        ]
        y_offset = frame.shape[0] - 120
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, y_offset + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Video Action Labeler', frame)
        
        if playing:
            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            current_frame = (current_frame + 1) % total_frames
        else:
            key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            playing = not playing
        elif key == ord('w'):
            # 标注为前进
            end_frame = min(current_frame + sequence_length - 1, total_frames - 1)
            if current_frame not in annotations:
                annotations[current_frame] = {}
            annotations[current_frame][end_frame] = ACTION_MAP['w']
            print(f"标注: 帧 {current_frame}-{end_frame} 为 前进(W)")
        elif key == ord('a'):
            end_frame = min(current_frame + sequence_length - 1, total_frames - 1)
            if current_frame not in annotations:
                annotations[current_frame] = {}
            annotations[current_frame][end_frame] = ACTION_MAP['a']
            print(f"标注: 帧 {current_frame}-{end_frame} 为 左移(A)")
        elif key == ord('s'):
            end_frame = min(current_frame + sequence_length - 1, total_frames - 1)
            if current_frame not in annotations:
                annotations[current_frame] = {}
            annotations[current_frame][end_frame] = ACTION_MAP['s']
            print(f"标注: 帧 {current_frame}-{end_frame} 为 后退(S)")
        elif key == ord('d'):
            end_frame = min(current_frame + sequence_length - 1, total_frames - 1)
            if current_frame not in annotations:
                annotations[current_frame] = {}
            annotations[current_frame][end_frame] = ACTION_MAP['d']
            print(f"标注: 帧 {current_frame}-{end_frame} 为 右移(D)")
        elif key == ord(' '):
            end_frame = min(current_frame + sequence_length - 1, total_frames - 1)
            if current_frame not in annotations:
                annotations[current_frame] = {}
            annotations[current_frame][end_frame] = ACTION_MAP[' ']
            print(f"标注: 帧 {current_frame}-{end_frame} 为 跳跃(空格)")
        elif key == ord('i'):
            end_frame = min(current_frame + sequence_length - 1, total_frames - 1)
            if current_frame not in annotations:
                annotations[current_frame] = {}
            annotations[current_frame][end_frame] = ACTION_MAP['idle']
            print(f"标注: 帧 {current_frame}-{end_frame} 为 静止(I)")
        elif key == 81 or key == 2:  # 左箭头
            current_frame = max(0, current_frame - 10)
        elif key == 83 or key == 3:  # 右箭头
            current_frame = min(total_frames - 1, current_frame + 10)
        elif key == ord('r'):
            # 重置当前帧的标注
            if current_frame in annotations:
                del annotations[current_frame]
                print(f"已重置帧 {current_frame} 的标注")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 保存标注
    save_annotations()
    print("\n标注完成!")


def main():
    parser = argparse.ArgumentParser(description='视频动作标注工具')
    parser.add_argument('--video', type=str, required=True, help='视频路径')
    parser.add_argument('--model', type=str, default='./models/yolov8n-pose.pt', help='YOLO pose模型路径')
    parser.add_argument('--output', type=str, default=None, help='输出JSON路径（默认: 视频名_action_labels.json）')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--seq_len', type=int, default=30, help='标注序列长度（帧数）')
    parser.add_argument('--roi', type=str, default=None, help='ROI区域，格式: x,y,w,h')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        raise FileNotFoundError(f'视频不存在: {args.video}')
    if not os.path.exists(args.model):
        raise FileNotFoundError(f'模型不存在: {args.model}')
    
    # 确定输出路径
    if args.output is None:
        video_name = Path(args.video).stem
        args.output = f"./action_labels/{video_name}_action_labels.json"
    
    # 解析ROI
    roi = None
    if args.roi:
        try:
            parts = args.roi.split(',')
            if len(parts) == 4:
                roi = tuple(int(p.strip()) for p in parts)
        except Exception as e:
            print(f"ROI解析失败: {e}，将使用全画面")
    
    label_video_interactive(
        args.video, 
        args.model, 
        args.output,
        args.conf,
        args.seq_len,
        roi
    )


if __name__ == '__main__':
    main()


