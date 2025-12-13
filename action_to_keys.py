#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从视频/摄像头识别动作，并将识别结果转换为W/A/S/D/空格按键，以在游戏里复刻动作。
"""

import argparse
import time
from collections import Counter, deque

import cv2
import numpy as np
import torch
from infer_action import draw_keypoints, select_best_keypoints
from ultralytics import YOLO
from action_classifier import (
    ActionClassifier,
    ActionClassifierGRU,
    ActionClassifierTransformer,
    ActionClassifierSTGCN,
)

try:
    from pynput import keyboard
except ImportError as exc:  # pragma: no cover - 环境依赖检查
    raise ImportError("需要安装pynput: pip install pynput") from exc


KEY_OUTPUT_MAP = {
    "w": "w",
    "a": "a",
    "s": "s",
    "d": "d",
    " ": keyboard.Key.space,
}


def load_action_model_any(model_path, device="cpu"):
    """加载动作模型，兼容 lstm/gru/transformer/stgcn。"""
    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint.get("model_type", "lstm")
    input_dim = checkpoint.get("input_dim", 34)
    hidden_dim = checkpoint.get("hidden_dim", 128)
    num_layers = checkpoint.get("num_layers", 2)
    num_classes = checkpoint.get("num_classes", 6)
    action_map = checkpoint.get("action_map", {})
    sequence_length = checkpoint.get("sequence_length", 30)

    if model_type == "lstm":
        model = ActionClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
        )
    elif model_type == "gru":
        model = ActionClassifierGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
        )
    elif model_type == "transformer":
        model = ActionClassifierTransformer(
            input_dim=input_dim,
            d_model=hidden_dim,
            num_classes=num_classes,
        )
    elif model_type == "stgcn":
        num_nodes = checkpoint.get("num_nodes", 17)
        in_channels = checkpoint.get("in_channels", 2)
        model = ActionClassifierSTGCN(
            num_nodes=num_nodes,
            in_channels=in_channels,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    action_names = {v: k for k, v in action_map.items()} if action_map else {}
    if not action_names:
        action_names = {
            0: "w",
            1: "a",
            2: "s",
            3: "d",
            4: "space",
            5: "idle",
        }
    return model, action_names, sequence_length, model_type


def normalize_action_name(name: str | None) -> str | None:
    """将模型输出的动作名规范化为按键映射可识别的形式。"""
    if name is None:
        return "idle"
    norm = str(name).strip().lower()
    if norm in {"idle", "none"}:
        return "idle"
    if norm in {"space", "jump"}:
        return " "
    if norm in KEY_OUTPUT_MAP:
        return norm
    return None


class KeyActuator:
    """负责将动作转换为键盘按下/抬起。"""

    def __init__(self, jump_hold: float = 0.08, jump_interval: float = 0.35):
        self.controller = keyboard.Controller()
        self.pressed_keys: set = set()
        self.jump_hold = jump_hold
        self.jump_interval = jump_interval
        self._last_jump_ts = 0.0

    def release_all(self):
        for k in list(self.pressed_keys):
            try:
                self.controller.release(k)
            except Exception:
                pass
        self.pressed_keys.clear()

    def apply(self, action_name: str | None):
        action = normalize_action_name(action_name)
        if action is None:
            # 未知动作，全部释放
            self.release_all()
            return

        if action == "idle":
            self.release_all()
            return

        # 空格视为跳跃：短按+节流
        if action == " ":
            now = time.time()
            if now - self._last_jump_ts >= self.jump_interval:
                self.release_all()
                key_obj = KEY_OUTPUT_MAP[action]
                try:
                    self.controller.press(key_obj)
                    time.sleep(self.jump_hold)
                    self.controller.release(key_obj)
                except Exception:
                    pass
                self._last_jump_ts = now
            return

        key_obj = KEY_OUTPUT_MAP.get(action)
        if key_obj is None:
            self.release_all()
            return

        # 若当前已按住同一键则不重复操作
        if self.pressed_keys == {key_obj}:
            return

        self.release_all()
        try:
            self.controller.press(key_obj)
            self.pressed_keys = {key_obj}
        except Exception:
            self.pressed_keys.clear()


def draw_key_ui(frame, action_text: str, conf: float, margin: int = 10):
    """在右上角绘制当前动作小面板。"""
    h, w = frame.shape[:2]
    line1 = action_text
    line2 = f"conf: {conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw1, th1), _ = cv2.getTextSize(line1, font, scale, thickness)
    (tw2, th2), _ = cv2.getTextSize(line2, font, scale, thickness)
    box_w = max(tw1, tw2) + 16
    box_h = th1 + th2 + 22
    x1 = w - box_w - margin
    y1 = margin
    x2 = x1 + box_w
    y2 = y1 + box_h

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # 边框
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

    # 文本
    cv2.putText(
        frame, line1, (x1 + 8, y1 + 16 + th1 // 2), font, scale, (0, 255, 0), thickness
    )
    cv2.putText(
        frame,
        line2,
        (x1 + 8, y1 + 16 + th1 + 12 + th2 // 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
    )

    return frame


def run_action_to_keys(
    source,
    pose_model_path: str,
    action_model_path: str,
    conf: float = 0.5,
    roi: tuple[int, int, int, int] | None = None,
    device: str = "cuda",
    smooth: int = 3,
    jump_hold: float = 0.08,
    jump_interval: float = 0.35,
    show_preview: bool = True,
    playback_speed: float = 1.0,
    stop_event=None,
):
    """主循环：推理动作并驱动按键。"""
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，自动切换为CPU")
        device = "cpu"

    print("加载pose模型...")
    pose_model = YOLO(pose_model_path)

    print("加载动作分类模型...")
    action_model, action_names, model_seq_len, model_type = load_action_model_any(
        action_model_path, device=device
    )
    sequence_length = model_seq_len
    print(f"动作类别: {action_names}")
    print(f"使用序列长度: {sequence_length}, 模型类型: {model_type}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    base_interval = 1.0 / fps if fps > 0 else 0.0
    if playback_speed <= 0:
        playback_speed = 1.0
    frame_interval = base_interval / playback_speed if base_interval > 0 else 0.0

    keypoint_buffer = deque(maxlen=sequence_length)
    action_history = deque(maxlen=max(1, smooth))
    actuator = KeyActuator(jump_hold=jump_hold, jump_interval=jump_interval)

    frame_idx = 0
    current_action = "idle"
    current_conf = 0.0
    smoothed_action = "idle"

    print("开始推理并发送按键...(按 Q/ESC 退出)")
    while True:
        if stop_event is not None and stop_event.is_set():
            break
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        h_full, w_full = frame.shape[:2]
        # ROI裁剪（可选）
        if roi is not None:
            x, y, w_roi, h_roi = roi
            if 0 <= x < w_full and 0 <= y < h_full:
                roi_frame = frame[y : y + h_roi, x : x + w_roi]
                results = pose_model(roi_frame, conf=conf, verbose=False)
            else:
                results = []
        else:
            results = pose_model(frame, conf=conf, verbose=False)

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

        if best_kp is not None:
            keypoint_buffer.append(best_kp.flatten())
        elif len(keypoint_buffer) > 0:
            keypoint_buffer.append(keypoint_buffer[-1])

        if len(keypoint_buffer) == sequence_length:
            seq_array = np.array(list(keypoint_buffer), dtype=np.float32)
            if model_type == "stgcn":
                seq_tensor = torch.FloatTensor(seq_array.reshape(1, sequence_length, 17, 2)).to(device)
            else:
                seq_tensor = torch.FloatTensor(seq_array).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = action_model(seq_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf_score, pred = torch.max(probs, 1)
                action_id = pred.item()
                current_conf = conf_score.item()
                current_action = action_names.get(action_id, str(action_id))
                action_history.append(action_id)

                # 平滑投票
                if len(action_history) >= 1:
                    smoothed_id = Counter(action_history).most_common(1)[0][0]
                    smoothed_action = action_names.get(smoothed_id, str(smoothed_id))
                else:
                    smoothed_action = current_action

                actuator.apply(smoothed_action)
        else:
            actuator.apply("idle")

        if show_preview:
            if best_kp is not None:
                frame = draw_keypoints(frame, best_kp)
            text_action = f"Action: {current_action} (p={current_conf:.2f})"
            cv2.putText(
                frame,
                text_action,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            frame = draw_key_ui(frame, f"{smoothed_action}", current_conf)
            cv2.putText(
                frame,
                "Press Q to quit",
                (10, h_full - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Action -> Keys", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # Q或ESC退出
                break

        frame_idx += 1

        # 控制播放速度（仅当有FPS信息时）
        if frame_interval > 0:
            elapsed = time.time() - frame_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    actuator.release_all()
    cap.release()
    cv2.destroyAllWindows()
    print("结束，已释放所有按键。")


def parse_roi(arg: str | None):
    if not arg:
        return None
    try:
        parts = [int(p.strip()) for p in arg.split(",")]
        if len(parts) == 4:
            return tuple(parts)
    except Exception:
        pass
    return None


def parse_source(arg: str):
    try:
        return int(arg)
    except ValueError:
        return arg


def main():
    parser = argparse.ArgumentParser(
        description="从视频/摄像头识别动作并自动按键到游戏"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="视频源，默认摄像头0；也可填写视频文件路径",
    )
    parser.add_argument(
        "--pose_model",
        type=str,
        default="./models/yolov8n-pose.pt",
        help="YOLO pose模型路径",
    )
    parser.add_argument(
        "--action_model", type=str, required=True, help="动作分类模型路径 (.pth)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="pose置信度阈值 (默认0.5)"
    )
    parser.add_argument("--roi", type=str, default=None, help="ROI区域 x,y,w,h")
    parser.add_argument(
        "--device", type=str, default="cuda", help="推理设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=3,
        help="动作投票窗口大小，越大越平滑但延迟越高",
    )
    parser.add_argument(
        "--jump_hold", type=float, default=0.08, help="空格按下保持时间(秒)"
    )
    parser.add_argument(
        "--jump_interval", type=float, default=0.35, help="空格触发最小间隔(秒)"
    )
    parser.add_argument(
        "--no_preview", action="store_true", help="不显示预览窗口，只输出按键"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="播放速度倍数，视频源有效 (默认1.0)"
    )

    args = parser.parse_args()
    roi = parse_roi(args.roi)
    source = parse_source(args.source)

    run_action_to_keys(
        source=source,
        pose_model_path=args.pose_model,
        action_model_path=args.action_model,
        conf=args.conf,
        roi=roi,
        device=args.device,
        smooth=args.smooth,
        jump_hold=args.jump_hold,
        jump_interval=args.jump_interval,
        show_preview=not args.no_preview,
        playback_speed=args.speed,
    )


if __name__ == "__main__":
    main()

