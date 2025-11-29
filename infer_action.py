#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作识别推理脚本
从视频中实时识别动作（W/A/S/D/空格跳跃）
"""

import os
import cv2
import numpy as np
import torch
import argparse
from collections import deque
from ultralytics import YOLO
from action_classifier import ActionClassifier, ActionClassifierGRU, ActionClassifierTransformer


def select_best_keypoints(result):
    """从单帧结果中选择一组关键点"""
    if not hasattr(result, 'keypoints') or result.keypoints is None or len(result.keypoints) == 0:
        return None
    kps = result.keypoints.xyn
    if kps is None or len(kps) == 0:
        return None

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


def load_action_model(model_path, device='cuda'):
    """加载动作分类模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    model_type = checkpoint.get('model_type', 'lstm')
    input_dim = checkpoint.get('input_dim', 34)
    hidden_dim = checkpoint.get('hidden_dim', 128)
    num_layers = checkpoint.get('num_layers', 2)
    num_classes = checkpoint.get('num_classes', 6)
    action_map = checkpoint.get('action_map', {})
    
    # 创建模型
    if model_type == 'lstm':
        model = ActionClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes
        )
    elif model_type == 'gru':
        model = ActionClassifierGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes
        )
    elif model_type == 'transformer':
        model = ActionClassifierTransformer(
            input_dim=input_dim,
            d_model=hidden_dim,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 创建动作名称映射
    action_names = {v: k for k, v in action_map.items()}
    
    return model, action_names, checkpoint.get('sequence_length', 30)


def draw_keypoints(frame, kp_2d):
    """在帧上绘制关键点"""
    h, w = frame.shape[:2]
    if kp_2d is None:
        return frame
    
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
    
    return frame


def infer_video(
    video_path,
    pose_model_path,
    action_model_path,
    conf_threshold=0.5,
    sequence_length=30,
    roi=None,
    device='cuda',
    output_path=None
):
    """对视频进行动作识别"""
    
    # 加载模型
    print("加载pose模型...")
    pose_model = YOLO(pose_model_path)
    
    print("加载动作分类模型...")
    action_model, action_names, model_seq_len = load_action_model(action_model_path, device)
    sequence_length = model_seq_len  # 使用模型训练时的序列长度
    print(f"模型序列长度: {sequence_length}")
    print(f"动作类别: {action_names}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频: {video_path}')
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {total_frames}帧, {fps:.2f}fps, {width}x{height}")
    
    # 视频写入器（如果指定输出路径）
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 关键点序列缓冲区
    keypoint_buffer = deque(maxlen=sequence_length)
    
    # 动作历史（用于平滑）
    action_history = deque(maxlen=5)
    
    frame_idx = 0
    current_action = "未识别"
    current_confidence = 0.0
    
    print("\n开始推理... (按Q退出)")
    
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
                    results = pose_model(roi_frame, conf=conf_threshold, verbose=False)
                else:
                    results = []
            else:
                results = []
        else:
            results = pose_model(frame, conf=conf_threshold, verbose=False)
        
        # 提取关键点
        best_kp = None
        for r in results:
            kp_local = select_best_keypoints(r)
            if kp_local is not None:
                if roi is not None:
                    x, y, w_roi, h_roi = roi
                    kp_global_x = (kp_local[:, 0] * w_roi + x) / w_full
                    kp_global_y = (kp_local[:, 1] * h_roi + y) / h_full
                    best_kp = np.stack([kp_global_x, kp_global_y], axis=1)
                else:
                    best_kp = kp_local
                break
        
        # 添加到缓冲区
        if best_kp is not None:
            keypoint_buffer.append(best_kp.flatten())
        else:
            # 如果没有检测到，使用上一帧（如果存在）
            if len(keypoint_buffer) > 0:
                keypoint_buffer.append(keypoint_buffer[-1])
        
        # 当缓冲区满时进行动作识别
        if len(keypoint_buffer) == sequence_length:
            # 准备输入
            seq_array = np.array(list(keypoint_buffer), dtype=np.float32)
            seq_tensor = torch.FloatTensor(seq_array).unsqueeze(0).to(device)  # (1, seq_len, 34)
            
            # 推理
            with torch.no_grad():
                outputs = action_model(seq_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
                action_id = predicted.item()
                confidence_score = confidence.item()
                current_action = action_names.get(action_id, f"class_{action_id}")
                current_confidence = confidence_score
                
                action_history.append(action_id)
        
        # 绘制关键点
        if best_kp is not None:
            frame = draw_keypoints(frame, best_kp)
        
        # 绘制ROI框
        if roi is not None:
            x, y, w_roi, h_roi = roi
            cv2.rectangle(frame, (x, y), (x + w_roi, y + h_roi), (255, 0, 0), 2)
        
        # 显示动作识别结果
        action_text = f"Action: {current_action}"
        conf_text = f"Confidence: {current_confidence:.2f}"
        
        # 使用投票机制平滑动作（如果历史足够）
        if len(action_history) >= 3:
            from collections import Counter
            most_common_action_id = Counter(action_history).most_common(1)[0][0]
            smoothed_action = action_names.get(most_common_action_id, f"class_{most_common_action_id}")
            action_text = f"Action: {smoothed_action} (smoothed)"
        
        cv2.putText(frame, action_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, conf_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示
        cv2.imshow('Action Recognition', frame)
        
        # 保存到输出视频
        if out:
            out.write(frame)
        
        # 按键控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"已处理 {frame_idx}/{total_frames} 帧")
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\n推理完成! 共处理 {frame_idx} 帧")
    if output_path:
        print(f"结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='视频动作识别推理')
    parser.add_argument('--video', type=str, required=True, help='视频路径')
    parser.add_argument('--pose_model', type=str, default='./models/yolov8n-pose.pt', 
                       help='YOLO pose模型路径')
    parser.add_argument('--action_model', type=str, required=True, 
                       help='动作分类模型路径 (.pth)')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--roi', type=str, default=None, help='ROI区域，格式: x,y,w,h')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--output', type=str, default=None, help='输出视频路径（可选）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        raise FileNotFoundError(f'视频不存在: {args.video}')
    if not os.path.exists(args.pose_model):
        raise FileNotFoundError(f'Pose模型不存在: {args.pose_model}')
    if not os.path.exists(args.action_model):
        raise FileNotFoundError(f'动作模型不存在: {args.action_model}')
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    # 解析ROI
    roi = None
    if args.roi:
        try:
            parts = args.roi.split(',')
            if len(parts) == 4:
                roi = tuple(int(p.strip()) for p in parts)
        except Exception as e:
            print(f"ROI解析失败: {e}，将使用全画面")
    
    infer_video(
        args.video,
        args.pose_model,
        args.action_model,
        args.conf,
        roi=roi,
        device=args.device,
        output_path=args.output
    )


if __name__ == '__main__':
    main()


