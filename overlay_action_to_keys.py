#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小悬浮窗：一键启动/停止动作识别→按键复刻
默认用摄像头0，可填写视频路径；按 ESC 或点击“停止”退出。
"""

import argparse
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from action_to_keys import run_action_to_keys


class OverlayApp:
    def __init__(self, root, default_source="0", default_pose="./models/yolov8n-pose.pt",
                 default_action="./models/action_classifier/best_stgcn_action_classifier.pth",
                 default_speed=1.0):
        self.root = root
        self.root.title("动作复刻")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)
        self.running_thread = None
        self.stop_event = None

        # 保存默认参数（来源于前端）
        self.source = default_source
        self.pose_model = default_pose
        self.action_model = default_action
        self.speed = default_speed

        pad = {"padx": 8, "pady": 6}

        ttk.Label(root, text="使用当前动作识别的输入：", foreground="#444").grid(row=0, column=0, sticky="w", **pad)
        ttk.Label(root, text=f"源: {self.source}").grid(row=1, column=0, sticky="w", **pad)
        ttk.Label(root, text=f"Pose: {self.pose_model}", wraplength=260, foreground="#666").grid(row=2, column=0, sticky="w", **pad)
        ttk.Label(root, text=f"Action: {self.action_model}", wraplength=260, foreground="#666").grid(row=3, column=0, sticky="w", **pad)
        ttk.Label(root, text=f"播放速度: x{self.speed}", foreground="#666").grid(row=4, column=0, sticky="w", **pad)

        btn_frame = ttk.Frame(root)
        btn_frame.grid(row=5, column=0, pady=6)
        self.start_btn = ttk.Button(btn_frame, text="开始复刻", command=self.start, width=12)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(btn_frame, text="停止", command=self.stop, state="disabled", width=8)
        self.stop_btn.pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="准备就绪，点击开始")
        ttk.Label(root, textvariable=self.status_var, foreground="#00897b").grid(row=6, column=0, sticky="w", **pad)
        ttk.Label(root, text="提示：切到游戏窗口，ESC 或“停止”结束", foreground="#777").grid(row=7, column=0, sticky="w", **pad)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Escape>", lambda _e: self.stop())

    def start(self):
        if self.running_thread and self.running_thread.is_alive():
            messagebox.showinfo("提示", "已在运行中")
            return
        try:
            source = self.parse_source(self.source)
        except Exception:
            messagebox.showerror("错误", "视频源有误")
            return
        pose_path = self.pose_model
        action_path = self.action_model
        if not pose_path or not action_path:
            messagebox.showerror("错误", "缺少模型路径")
            return

        self.stop_event = threading.Event()
        self.running_thread = threading.Thread(
            target=self.worker,
            args=(source, pose_path, action_path, self.speed, self.stop_event),
            daemon=True,
        )
        self.running_thread.start()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("运行中，切到游戏窗口...")

    def stop(self):
        if self.stop_event:
            self.stop_event.set()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def worker(self, source, pose_model, action_model, speed, stop_event, show_preview=False):
        try:
            run_action_to_keys(
                source=source,
                pose_model_path=pose_model,
                action_model_path=action_model,
                show_preview=show_preview,
                playback_speed=speed,
                stop_event=stop_event,
            )
        except Exception as e:
            msg = f"运行异常: {e}"
            self.status_var.set(msg)
            try:
                messagebox.showerror("运行异常", msg)
            except Exception:
                pass
        finally:
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            if stop_event:
                stop_event.set()

    @staticmethod
    def parse_source(val: str):
        try:
            return int(val)
        except ValueError:
            return val

    def on_close(self):
        self.stop()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="动作复刻悬浮窗")
    parser.add_argument("--source", type=str, default="0", help="视频源/摄像头编号")
    parser.add_argument("--pose_model", type=str, default="./models/yolov8n-pose.pt", help="Pose模型路径")
    parser.add_argument("--action_model", type=str, default="./models/action_classifier/best_stgcn_action_classifier.pth", help="动作识别模型路径")
    parser.add_argument("--preview", action="store_true", default=False, help="(已关闭预览，仅保留按键)")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍数（视频源）")
    args = parser.parse_args()

    root = tk.Tk()
    OverlayApp(
        root,
        default_source=args.source,
        default_pose=args.pose_model,
        default_action=args.action_model,
        default_speed=args.speed,
    )
    root.mainloop()


if __name__ == "__main__":
    main()

