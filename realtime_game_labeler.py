#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时游戏画面动作标注工具
捕获游戏画面，实时检测pose，通过键盘输入（W/A/S/D/空格）自动标注动作
"""

import os
import cv2
import json
import numpy as np
import argparse
import time
import threading
from pathlib import Path
from collections import deque
from datetime import datetime
from ultralytics import YOLO

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("警告: mss未安装，将使用PIL进行屏幕捕获。安装mss可提升性能: pip install mss")

try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("警告: pynput未安装，无法监听键盘。请安装: pip install pynput")

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


class ScreenCapture:
    """屏幕捕获类"""
    
    def __init__(self, monitor=None, use_mss=True):
        """
        Args:
            monitor: 屏幕区域，格式 (x, y, width, height) 或 None（全屏）
            use_mss: 是否使用mss库（更快）
        """
        self.monitor = monitor
        self.use_mss = use_mss and MSS_AVAILABLE
        
        if self.use_mss:
            self.sct = mss.mss()
            if monitor:
                self.monitor_dict = {
                    "top": monitor[1],
                    "left": monitor[0],
                    "width": monitor[2],
                    "height": monitor[3]
                }
            else:
                # 使用主显示器
                self.monitor_dict = self.sct.monitors[1]  # monitors[0]是所有显示器，[1]是主显示器
        else:
            try:
                from PIL import ImageGrab
                self.ImageGrab = ImageGrab
            except ImportError:
                raise ImportError("需要安装PIL: pip install pillow")
    
    def capture(self):
        """捕获一帧屏幕"""
        if self.use_mss:
            sct_img = self.sct.grab(self.monitor_dict)
            # 转换为numpy数组
            img = np.array(sct_img)
            # mss返回BGRA格式，转换为BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            if self.monitor:
                bbox = (self.monitor[0], self.monitor[1], 
                       self.monitor[0] + self.monitor[2], 
                       self.monitor[1] + self.monitor[3])
            else:
                bbox = None
            img = self.ImageGrab.grab(bbox=bbox)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img


class KeyboardListener:
    """键盘监听类"""
    
    def __init__(self):
        self.current_keys = set()
        self.lock = threading.Lock()
        self.listener = None
        
        if not KEYBOARD_AVAILABLE:
            raise ImportError("需要安装pynput: pip install pynput")
    
    def on_press(self, key):
        """按键按下"""
        try:
            key_char = key.char.lower() if hasattr(key, 'char') and key.char else None
            if key_char in ['w', 'a', 's', 'd']:
                with self.lock:
                    self.current_keys.add(key_char)
            elif key == keyboard.Key.space:
                with self.lock:
                    self.current_keys.add(' ')
        except AttributeError:
            pass
    
    def on_release(self, key):
        """按键释放"""
        try:
            key_char = key.char.lower() if hasattr(key, 'char') and key.char else None
            if key_char in ['w', 'a', 's', 'd']:
                with self.lock:
                    self.current_keys.discard(key_char)
            elif key == keyboard.Key.space:
                with self.lock:
                    self.current_keys.discard(' ')
            elif key == keyboard.Key.esc:
                # ESC键用于停止
                return False
        except AttributeError:
            pass
    
    def start(self):
        """启动监听"""
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
    
    def stop(self):
        """停止监听"""
        if self.listener:
            self.listener.stop()
    
    def get_current_action(self):
        """获取当前动作（优先级: W > A > S > D > 空格）"""
        with self.lock:
            keys = self.current_keys.copy()
        
        if not keys:
            return 'idle'
        
        # 优先级顺序
        priority = ['w', 'a', 's', 'd', ' ']
        for key in priority:
            if key in keys:
                return key
        
        return 'idle'


class RealtimeGameLabeler:
    """实时游戏标注器"""
    
    def __init__(self, model_path, monitor=None, conf_threshold=0.5, 
                 sequence_length=30, output_path=None, fps=30):
        """
        Args:
            model_path: YOLO pose模型路径
            monitor: 屏幕区域 (x, y, width, height) 或 None（全屏）
            conf_threshold: 置信度阈值
            sequence_length: 序列长度（帧数）
            output_path: 输出JSON路径
            fps: 捕获帧率
        """
        self.model_path = model_path
        self.monitor = monitor
        self.conf_threshold = conf_threshold
        self.sequence_length = sequence_length
        self.fps = fps
        self.frame_interval = 1.0 / fps
        
        # 确定输出路径
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"game_action_labels_{timestamp}.json"
        self.output_path = output_path
        
        # 初始化组件
        print("初始化屏幕捕获...")
        self.screen_capture = ScreenCapture(monitor=monitor)
        
        print("加载pose模型...")
        self.pose_model = YOLO(model_path)
        
        print("初始化键盘监听...")
        self.keyboard_listener = KeyboardListener()
        
        # 数据存储
        self.keypoint_sequence = []
        self.action_sequence = []  # 每帧对应的动作
        self.frame_timestamps = []
        
        # 运行状态
        self.running = False
        self.stop_flag = False
        
        print(f"输出文件: {self.output_path}")
        print("\n操作说明:")
        print("  W/A/S/D/空格 - 标注动作（按住按键）")
        print("  ESC - 停止标注并保存")
        print("=" * 60)
    
    def run(self):
        """运行标注"""
        self.running = True
        self.stop_flag = False
        
        # 启动键盘监听
        self.keyboard_listener.start()
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while not self.stop_flag:
                frame_start = time.time()
                
                # 捕获屏幕
                frame = self.screen_capture.capture()
                
                # 检测pose
                results = self.pose_model(frame, conf=self.conf_threshold, verbose=False)
                
                # 提取关键点
                best_kp = None
                for r in results:
                    kp = select_best_keypoints(r)
                    if kp is not None:
                        best_kp = kp
                        break
                
                # 获取当前动作
                current_action = self.keyboard_listener.get_current_action()
                action_id = ACTION_MAP.get(current_action, 5)
                
                # 保存数据
                if best_kp is not None:
                    self.keypoint_sequence.append(best_kp.flatten().tolist())
                else:
                    # 如果没有检测到，使用上一帧（如果存在）
                    if len(self.keypoint_sequence) > 0:
                        self.keypoint_sequence.append(self.keypoint_sequence[-1])
                    else:
                        # 如果完全没有数据，跳过
                        continue
                
                self.action_sequence.append(action_id)
                self.frame_timestamps.append(time.time())
                
                # 显示预览（可选）
                display_frame = frame.copy()
                if best_kp is not None:
                    h, w = display_frame.shape[:2]
                    kp = np.array(best_kp).reshape(17, 2)
                    pts_px = (kp * np.array([w, h])).astype(int)
                    
                    # 绘制关键点
                    for (x, y) in pts_px:
                        cv2.circle(display_frame, (int(x), int(y)), 3, (0, 255, 255), -1)
                    
                    # 绘制骨架
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
                            cv2.line(display_frame, (int(ax), int(ay)), 
                                   (int(bx), int(by)), (0, 200, 255), 2)
                
                # 显示动作信息
                action_text = f"Action: {current_action.upper() if current_action != 'idle' else 'IDLE'}"
                cv2.putText(display_frame, action_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Frames: {frame_count}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "Press ESC to stop", (10, display_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 缩小显示（如果太大）
                display_h, display_w = display_frame.shape[:2]
                max_display_size = 800
                if max(display_h, display_w) > max_display_size:
                    scale = max_display_size / max(display_h, display_w)
                    new_w = int(display_w * scale)
                    new_h = int(display_h * scale)
                    display_frame = cv2.resize(display_frame, (new_w, new_h))
                
                cv2.imshow('Realtime Game Labeler', display_frame)
                
                # 检查ESC键（通过OpenCV窗口）
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    self.stop_flag = True
                    break
                
                frame_count += 1
                
                # 控制帧率
                elapsed = time.time() - frame_start
                sleep_time = max(0, self.frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # 每100帧打印一次进度
                if frame_count % 100 == 0:
                    elapsed_total = time.time() - start_time
                    actual_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
                    print(f"已标注 {frame_count} 帧, 实际FPS: {actual_fps:.1f}, "
                          f"当前动作: {current_action}")
        
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.running = False
            self.keyboard_listener.stop()
            cv2.destroyAllWindows()
            
            # 保存数据
            self.save_annotations()
    
    def save_annotations(self):
        """保存标注数据"""
        if len(self.keypoint_sequence) == 0:
            print("没有数据可保存")
            return
        
        print(f"\n保存标注数据... (共 {len(self.keypoint_sequence)} 帧)")
        
        # 将动作序列转换为标注格式
        annotations = {}
        current_action = None
        start_frame = None
        
        for i, action_id in enumerate(self.action_sequence):
            if action_id != current_action:
                # 动作改变，保存之前的片段
                if current_action is not None and start_frame is not None:
                    end_frame = i - 1
                    if start_frame not in annotations:
                        annotations[start_frame] = {}
                    annotations[start_frame][end_frame] = current_action
                
                # 开始新动作
                current_action = action_id
                start_frame = i
        
        # 保存最后一个片段
        if current_action is not None and start_frame is not None:
            end_frame = len(self.action_sequence) - 1
            if start_frame not in annotations:
                annotations[start_frame] = {}
            annotations[start_frame][end_frame] = current_action
        
        # 计算实际FPS
        if len(self.frame_timestamps) > 1:
            time_diffs = [self.frame_timestamps[i+1] - self.frame_timestamps[i] 
                         for i in range(len(self.frame_timestamps)-1)]
            avg_fps = 1.0 / np.mean(time_diffs) if np.mean(time_diffs) > 0 else self.fps
        else:
            avg_fps = self.fps
        
        data = {
            'source': 'realtime_game_labeler',
            'total_frames': len(self.keypoint_sequence),
            'fps': avg_fps,
            'sequence_length': self.sequence_length,
            'keypoint_sequence': self.keypoint_sequence,
            'frame_indices': list(range(len(self.keypoint_sequence))),
            'annotations': {str(k): {str(ek): ev for ek, ev in v.items()} 
                          for k, v in annotations.items()},
            'action_map': ACTION_MAP,
            'monitor': self.monitor,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 标注已保存到: {self.output_path}")
        print(f"  总帧数: {len(self.keypoint_sequence)}")
        print(f"  标注片段数: {len(annotations)}")
        
        # 统计动作分布
        action_counts = {}
        for action_id in self.action_sequence:
            action_name = ACTION_NAMES.get(action_id, f"class_{action_id}")
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        
        print("\n动作分布:")
        for action_name, count in sorted(action_counts.items()):
            percentage = count / len(self.action_sequence) * 100
            print(f"  {action_name}: {count} 帧 ({percentage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='实时游戏画面动作标注工具')
    parser.add_argument('--model', type=str, default='./models/yolov8n-pose.pt', 
                       help='YOLO pose模型路径')
    parser.add_argument('--monitor', type=str, default=None, 
                       help='屏幕区域，格式: x,y,width,height (例如: 0,0,1920,1080)，不指定则全屏')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--seq_len', type=int, default=30, help='序列长度（用于后续训练）')
    parser.add_argument('--fps', type=float, default=30, help='捕获帧率')
    parser.add_argument('--output', type=str, default=None, 
                       help='输出JSON路径（默认: game_action_labels_时间戳.json）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        raise FileNotFoundError(f'模型不存在: {args.model}')
    
    if not KEYBOARD_AVAILABLE:
        raise ImportError("需要安装pynput: pip install pynput")
    
    # 解析monitor
    monitor = None
    if args.monitor:
        try:
            parts = args.monitor.split(',')
            if len(parts) == 4:
                monitor = tuple(int(p.strip()) for p in parts)
        except Exception as e:
            print(f"Monitor解析失败: {e}，将使用全屏")
    
    # 创建标注器
    labeler = RealtimeGameLabeler(
        model_path=args.model,
        monitor=monitor,
        conf_threshold=args.conf,
        sequence_length=args.seq_len,
        output_path=args.output,
        fps=args.fps
    )
    
    # 运行
    print("\n3秒后开始标注...")
    print("请切换到游戏窗口，然后按住W/A/S/D/空格进行标注")
    time.sleep(3)
    
    labeler.run()


if __name__ == '__main__':
    main()


