#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立测试脚本：
1) 读取视频
2) YOLO Pose 提取17点关键点
3) 使用 Open3D 实时渲染可旋转的3D火柴人（将2D关键点投射到z=0平面）

依赖：
  pip install ultralytics opencv-python open3d

用法示例：
  python test_pose3d.py --video ./datasets/videos/demo.mp4 --model ./models/yolov8n-pose.pt --conf 0.5

测试结果：
  人物关键点能完好映射到火柴人，但是火柴人仅能体现二维，例如人物前进不能体现手臂前伸，只能看出手臂弯曲

"""

import argparse
import time
import os
import numpy as np
import cv2
import open3d as o3d
from ultralytics import YOLO


# COCO 17关键点骨架连线（与前端一致）
# COCO 17点索引：0鼻，1左眼，2右眼，3左耳，4右耳，5左肩，6右肩，7左肘，8右肘，9左腕，10右腕，11左髋，12右髋，13左膝，14右膝，15左踝，16右踝
# 增加头部连线，便于显示“头部”轮廓和与肩部的连接
COCO_EDGES = [
    # 头部
    [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6],
    # 上肢
    [5, 7], [7, 9],
    [6, 8], [8, 10],
    # 下肢
    [11, 13], [13, 15],
    [12, 14], [14, 16],
    # 身躯
    [5, 6],
    [11, 12],
    [5, 11], [6, 12]
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='./1.mp4', help='视频路径，如 ./datasets/videos/demo.mp4')
    parser.add_argument('--model', type=str, default='./models/yolov8n-pose.pt', help='YOLO pose 模型路径')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--scale', type=float, default=1.0, help='3D视图缩放（越大越占画面）')
    parser.add_argument('--headless', action='store_true', help='仅推理不渲染（用于性能测试）')
    parser.add_argument('--roi', type=str, default=None, help='预设ROI区域，格式: x,y,w,h (例如: 100,100,400,300)，不指定则在第一帧手动选择')
    return parser.parse_args()


def select_best_keypoints(result) -> np.ndarray:
    """从单帧结果中选择一组关键点（优先置信度最高的目标）。
    返回形状 (17, 2) 的归一化坐标数组，若无则返回 None。
    """
    if not hasattr(result, 'keypoints') or result.keypoints is None or len(result.keypoints) == 0:
        return None
    kps = result.keypoints.xyn  # 归一化关键点 (num, 17, 2)
    if kps is None or len(kps) == 0:
        return None

    # 若有 boxes.conf 则按置信度排序选第一
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


def build_lineset():
    # 初始点（17个）全0
    points = np.zeros((17, 3), dtype=np.float64)
    # 线段索引
    lines = np.array(COCO_EDGES, dtype=np.int32)
    # 统一颜色
    colors = np.tile(np.array([[0.27, 0.8, 1.0]], dtype=np.float64), (lines.shape[0], 1))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def map_2d_to_3d(kp_2d: np.ndarray, scale: float = 1.6) -> np.ndarray:
    """将归一化的2D关键点 (17,2) 映射至3D平面坐标 (17,3)，中心化并翻转y。"""
    cx, cy = 0.5, 0.5
    x = (kp_2d[:, 0] - cx) * scale
    y = -(kp_2d[:, 1] - cy) * scale
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1)

def map_2d_to_3d_centered(kp_2d: np.ndarray, center_xy: np.ndarray, scale: float) -> np.ndarray:
    """将归一化2D关键点按给定中心对齐至画面中心(0.5,0.5)，再投射到z=0平面。"""
    x = (kp_2d[:, 0] - center_xy[0]) * scale
    y = -(kp_2d[:, 1] - center_xy[1]) * scale
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1)


def main():
    args = parse_args()
    if not os.path.exists(args.video):
        raise FileNotFoundError(f'视频不存在: {args.video}')
    if not os.path.exists(args.model):
        raise FileNotFoundError(f'模型不存在: {args.model}')

    print('加载模型中...')
    model = YOLO(args.model)
    print('模型加载完成')

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError('无法打开视频')

    # ========== ROI选择 ==========
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError('无法读取视频第一帧')
    
    h_full, w_full = first_frame.shape[:2]
    roi = None
    
    if args.roi:
        # 从命令行参数解析ROI
        try:
            parts = args.roi.split(',')
            if len(parts) == 4:
                x, y, w, h = [int(p.strip()) for p in parts]
                roi = (max(0, x), max(0, y), min(w, w_full - x), min(h, h_full - y))
                print(f'使用预设ROI: {roi}')
            else:
                raise ValueError('ROI格式错误，应为 x,y,w,h')
        except Exception as e:
            print(f'解析ROI参数失败: {e}，将在第一帧手动选择')
            args.roi = None
    
    if not args.headless and roi is None:
        # 在第一帧手动选择ROI
        print('请在视频窗口上拖动鼠标选择ROI区域，按ENTER确认，按ESC取消')
        print('提示：选择包含人物的区域可提高识别准确度和速度')
        # 使用简单的窗口名避免编码问题
        window_name = 'Select ROI'
        try:
            # 先创建命名窗口
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, first_frame)
            cv2.waitKey(100)  # 等待窗口完全渲染
            # 尝试移动窗口位置
            try:
                cv2.moveWindow(window_name, 20, 50)
            except Exception:
                pass
            # 调用selectROI
            roi = cv2.selectROI(window_name, first_frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(window_name)
        except Exception as e:
            print(f'ROI选择窗口创建失败: {e}，尝试使用默认方法')
            try:
                roi = cv2.selectROI(first_frame, showCrosshair=True, fromCenter=False)
            except Exception as e2:
                print(f'ROI选择失败: {e2}，将使用全画面')
                roi = None
        
        if roi is not None and (roi[2] <= 0 or roi[3] <= 0):
            print('未选择ROI或选择无效，将使用全画面')
            roi = None
        elif roi is not None:
            print(f'已选择ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}')
    elif args.headless and roi is None:
        print('Headless模式且未指定--roi，将使用全画面进行识别')
    
    # 重置视频到开头
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    vis = None
    line_set = None
    if not args.headless:
        vis = o3d.visualization.Visualizer()
        # 右侧放置Open3D窗口（尽量与视频窗口并排）
        vis.create_window(window_name='Pose3D - Stick Figure', width=960, height=720, left=1000, top=50, visible=True)
        line_set = build_lineset()
        vis.add_geometry(line_set)

        # 简单光照+坐标系
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis.add_geometry(coord)

        # 相机初始位置
        ctr = vis.get_view_control()
        # 让相机拉远一点看全身
        ctr.set_zoom(2.7)

    t0 = time.time()
    frame_count = 0
    # 居中EMA（根据识别框中心）
    center_ema = np.array([0.5, 0.5], dtype=np.float64)
    alpha = 0.2  # 平滑系数，越大越快贴合
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('视频结束')
                break

            h_full, w_full = frame.shape[:2]
            
            # 在ROI区域内推理
            if roi is not None:
                x, y, w_roi, h_roi = roi
                # 裁剪ROI区域
                roi_frame = frame[y:y+h_roi, x:x+w_roi]
                if roi_frame.size == 0:
                    print(f'警告: ROI区域无效，跳过本帧')
                    continue
                # 在ROI区域推理
                results = model(roi_frame, conf=args.conf)
            else:
                # 全画面推理
                results = model(frame, conf=args.conf)
            
            # 取单帧结果对象（ultralytics返回list）
            best_kp = None
            for r in results:
                kp_local = select_best_keypoints(r)
                if kp_local is not None:
                    # 如果使用了ROI，将关键点从ROI局部坐标转换为原图全局归一化坐标
                    if roi is not None:
                        x, y, w_roi, h_roi = roi
                        # kp_local是相对于ROI的归一化坐标 [0,1]
                        # 转换为原图的归一化坐标
                        kp_global_x = (kp_local[:, 0] * w_roi + x) / w_full
                        kp_global_y = (kp_local[:, 1] * h_roi + y) / h_full
                        best_kp = np.stack([kp_global_x, kp_global_y], axis=1)
                    else:
                        best_kp = kp_local
                    break

            if not args.headless:
                # 在视频帧上叠加2D关键点与骨架（像素坐标）
                h, w = frame.shape[:2]
                # 绘制ROI框
                if roi is not None:
                    x_roi, y_roi, w_roi, h_roi = roi
                    cv2.rectangle(frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (255, 0, 0), 2)
                    cv2.putText(frame, 'ROI', (x_roi, max(y_roi - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                if best_kp is not None:
                    # 计算基于关键点的包围框中心（归一化坐标）
                    x_min = float(np.clip(best_kp[:, 0].min(), 0.0, 1.0))
                    x_max = float(np.clip(best_kp[:, 0].max(), 0.0, 1.0))
                    y_min = float(np.clip(best_kp[:, 1].min(), 0.0, 1.0))
                    y_max = float(np.clip(best_kp[:, 1].max(), 0.0, 1.0))
                    bbox_center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0], dtype=np.float64)
                    # EMA 平滑中心
                    center_ema = (1.0 - alpha) * center_ema + alpha * bbox_center

                    pts_px = np.stack([best_kp[:, 0] * w, best_kp[:, 1] * h], axis=1).astype(int)
                    # 画点
                    for (x, y) in pts_px:
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
                    # 画线
                    for e in COCO_EDGES:
                        a, b = e[0], e[1]
                        if 0 <= a < len(pts_px) and 0 <= b < len(pts_px):
                            ax, ay = pts_px[a]
                            bx, by = pts_px[b]
                            cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), (0, 200, 255), 2)

                    # 画包围盒与中心点（像素）
                    cv2.rectangle(frame,
                                  (int(x_min * w), int(y_min * h)),
                                  (int(x_max * w), int(y_max * h)),
                                  (0, 128, 255), 2)
                    cx_px = int(center_ema[0] * w)
                    cy_px = int(center_ema[1] * h)
                    cv2.drawMarker(frame, (cx_px, cy_px), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

                if best_kp is not None and line_set is not None:
                    # 以识别框中心为参考，将火柴人对齐到画面中心
                    pts3d = map_2d_to_3d_centered(best_kp, center_xy=center_ema, scale=args.scale)
                    line_set.points = o3d.utility.Vector3dVector(pts3d)
                    vis.update_geometry(line_set)
                else:
                    # 无人则清零（保持渲染）
                    zero_pts = np.zeros((17, 3), dtype=np.float64)
                    line_set.points = o3d.utility.Vector3dVector(zero_pts)
                    vis.update_geometry(line_set)

                vis.poll_events()
                vis.update_renderer()

                # 左侧显示视频窗口，尽量并排
                cv2.imshow('Video', frame)
                try:
                    cv2.moveWindow('Video', 20, 50)
                except Exception:
                    pass

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print('用户中断')
                    break

            frame_count += 1
    finally:
        cap.release()
        if vis is not None:
            vis.destroy_window()
        cv2.destroyAllWindows()

    dt = time.time() - t0
    if dt > 0:
        print(f'平均FPS(包含推理+渲染): {frame_count / dt:.2f}')


if __name__ == '__main__':
    main()


