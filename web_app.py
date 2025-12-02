#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MotionSync - 位姿提取与动作估计软件
通过浏览器访问的Web界面
"""

import os
import sys
import json
import logging
import subprocess
import time
import queue
import numpy as np
import cv2
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
import threading
import base64

# 导入现有功能模块
from auto_label import auto_label_yolo_format
from txt_coco_json import txt_to_coco, coco_to_txt
from txt_json import yolo_pose_txt_to_labelme, labelme_to_yolo_pose
from train_pose import train_yolov8_pose

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

app = Flask(__name__)

# 全局变量存储任务状态
task_status = {
    'running': False,
    'progress': 0,
    'message': '',
    'logs': []
}

# 视频关键点流状态
pose_stream_state = {
    'running': False,
    'stop_flag': False,
    'thread': None,
    'queue': None,
    'model': None,
    'action_model': None
}

# 动作标注状态
action_labeling_state = {
    'video_labeling': {
        'running': False,
        'thread': None,
        'output_path': None
    },
    'realtime_labeling': {
        'running': False,
        'thread': None,
        'stop_flag': False,
        'output_path': None
    }
}

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('web_app.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

class WebLogHandler(logging.Handler):
    """Web日志处理器"""
    def emit(self, record):
        log_entry = self.format(record)
        task_status['logs'].append(log_entry)
        # 保持最近100条日志
        if len(task_status['logs']) > 100:
            task_status['logs'] = task_status['logs'][-100:]

# 添加Web日志处理器
web_handler = WebLogHandler()
web_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(web_handler)

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/dataset_info')
def dataset_info():
    """获取数据集信息"""
    try:
        dataset_path = "./datasets"
        info = {
            'train_images': 0,
            'val_images': 0,
            'train_labels': 0,
            'val_labels': 0,
            'total_images': 0,
            'total_labels': 0
        }
        
        if os.path.exists(dataset_path):
            # 统计图片数量
            train_img_path = os.path.join(dataset_path, "images", "train")
            val_img_path = os.path.join(dataset_path, "images", "val")
            
            info['train_images'] = len([f for f in os.listdir(train_img_path) if f.lower().endswith(('.jpg', '.png'))]) if os.path.exists(train_img_path) else 0
            info['val_images'] = len([f for f in os.listdir(val_img_path) if f.lower().endswith(('.jpg', '.png'))]) if os.path.exists(val_img_path) else 0
            
            # 统计标签数量
            train_label_path = os.path.join(dataset_path, "labels", "train")
            val_label_path = os.path.join(dataset_path, "labels", "val")
            
            info['train_labels'] = len([f for f in os.listdir(train_label_path) if f.endswith('.txt')]) if os.path.exists(train_label_path) else 0
            info['val_labels'] = len([f for f in os.listdir(val_label_path) if f.endswith('.txt')]) if os.path.exists(val_label_path) else 0
            
            info['total_images'] = info['train_images'] + info['val_images']
            info['total_labels'] = info['train_labels'] + info['val_labels']
        
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auto_label', methods=['POST'])
def auto_label():
    """自动标定API"""
    global task_status
    
    if task_status['running']:
        return jsonify({'error': '任务正在运行中'}), 400
    
    try:
        data = request.json
        img_dir = data.get('img_dir', './datasets/images/train')
        model_path = data.get('model_path', './models/yolov8n-pose.pt')
        output_dir = data.get('output_dir', './datasets/labels/train')
        conf_threshold = float(data.get('conf_threshold', 0.5))
        
        # 验证参数
        if not os.path.exists(img_dir):
            return jsonify({'error': '图片目录不存在'}), 400
            
        if not os.path.exists(model_path):
            return jsonify({'error': '模型文件不存在'}), 400
        
        # 启动后台任务
        def run_auto_label():
            global task_status
            task_status['running'] = True
            task_status['progress'] = 0
            task_status['message'] = '开始自动标定...'
            task_status['logs'] = []
            
            try:
                logging.info(f"开始自动标定: 图片目录={img_dir}, 模型={model_path}, 输出目录={output_dir}")
                
                auto_label_yolo_format(img_dir, model_path, output_dir, conf_threshold)
                
                task_status['progress'] = 100
                task_status['message'] = '自动标定完成！'
                logging.info("自动标定完成！")
                
            except Exception as e:
                task_status['message'] = f'自动标定失败: {str(e)}'
                logging.error(f"自动标定失败: {str(e)}")
            finally:
                task_status['running'] = False
        
        # 在后台线程中运行
        thread = threading.Thread(target=run_auto_label)
        thread.start()
        
        return jsonify({'message': '自动标定任务已启动'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/convert', methods=['POST'])
def convert():
    """格式转换API"""
    global task_status
    
    if task_status['running']:
        return jsonify({'error': '任务正在运行中'}), 400
    
    try:
        data = request.json
        mode = data.get('mode')
        input_path = data.get('input_path', './datasets/labels/train')
        output_path = data.get('output_path', './datasets/labels_json/1.json')
        img_dir = data.get('img_dir', './datasets/images/train')
        
        # 验证参数
        if not mode or not input_path or not output_path:
            return jsonify({'error': '缺少必要参数'}), 400
        
        # 启动后台任务
        def run_convert():
            global task_status
            task_status['running'] = True
            task_status['progress'] = 0
            task_status['message'] = f'开始格式转换: {mode}'
            task_status['logs'] = []
            
            try:
                logging.info(f"开始格式转换: {mode}")
                
                if mode == "txt2coco":
                    txt_to_coco(img_dir, input_path, output_path)
                elif mode == "coco2txt":
                    coco_to_txt(input_path, output_path)
                elif mode == "txt2labelme":
                    yolo_pose_txt_to_labelme(img_dir, input_path, output_path)
                elif mode == "labelme2txt":
                    labelme_to_yolo_pose(input_path, output_path, img_dir)
                
                task_status['progress'] = 100
                task_status['message'] = '格式转换完成！'
                logging.info("格式转换完成！")
                
            except Exception as e:
                task_status['message'] = f'格式转换失败: {str(e)}'
                logging.error(f"格式转换失败: {str(e)}")
            finally:
                task_status['running'] = False
        
        # 在后台线程中运行
        thread = threading.Thread(target=run_convert)
        thread.start()
        
        return jsonify({'message': '格式转换任务已启动'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """获取任务状态"""
    return jsonify(task_status)

@app.route('/api/logs')
def logs():
    """获取日志"""
    return jsonify({'logs': task_status['logs']})

@app.route('/api/coco_annotator_status')
def coco_annotator_status():
    """获取COCO Annotator状态"""
    try:
        # 检查Docker是否运行
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({'running': False, 'error': 'Docker未运行'})
        
        # 检查COCO Annotator容器
        lines = result.stdout.split('\n')
        coco_running = any('coco-annotator' in line or '5000' in line for line in lines)
        
        return jsonify({
            'running': coco_running,
            'url': 'http://localhost:5000' if coco_running else None
        })
    except Exception as e:
        return jsonify({'running': False, 'error': str(e)})

@app.route('/api/start_coco_annotator', methods=['POST'])
def start_coco_annotator():
    """启动COCO Annotator"""
    try:
        coco_dir = os.path.join(os.getcwd(), 'coco-annotator-master')
        if not os.path.exists(coco_dir):
            return jsonify({'error': 'COCO Annotator目录不存在'}), 400
        
        # 启动COCO Annotator
        def start_coco():
            try:
                result = subprocess.run(
                    ['docker-compose', 'up', '-d'],
                    cwd=coco_dir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    logging.info("COCO Annotator启动成功")
                else:
                    logging.error(f"COCO Annotator启动失败: {result.stderr}")
            except Exception as e:
                logging.error(f"启动COCO Annotator时出错: {e}")
        
        # 在后台线程中启动
        thread = threading.Thread(target=start_coco)
        thread.start()
        
        return jsonify({'message': 'COCO Annotator正在启动中...'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_coco_annotator', methods=['POST'])
def stop_coco_annotator():
    """停止COCO Annotator"""
    try:
        coco_dir = os.path.join(os.getcwd(), 'coco-annotator-master')
        if not os.path.exists(coco_dir):
            return jsonify({'error': 'COCO Annotator目录不存在'}), 400
        
        # 停止COCO Annotator
        result = subprocess.run(
            ['docker-compose', 'down'],
            cwd=coco_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            logging.info("COCO Annotator已停止")
            return jsonify({'message': 'COCO Annotator已停止'})
        else:
            return jsonify({'error': f'停止失败: {result.stderr}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate_json', methods=['POST'])
def validate_json():
    """验证JSON文件"""
    try:
        data = request.json
        json_path = data.get('json_path')
        
        if not json_path or not os.path.exists(json_path):
            return jsonify({'error': 'JSON文件不存在'}), 400
        
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 验证JSON结构
        validation_result = {
            'valid': True,
            'images_count': len(json_data.get('images', [])),
            'annotations_count': len(json_data.get('annotations', [])),
            'categories_count': len(json_data.get('categories', [])),
            'issues': []
        }
        
        # 检查必要字段
        required_fields = ['images', 'annotations', 'categories']
        for field in required_fields:
            if field not in json_data:
                validation_result['issues'].append(f'缺少必要字段: {field}')
                validation_result['valid'] = False
        
        # 检查图片和标注的对应关系
        if 'images' in json_data and 'annotations' in json_data:
            image_ids = {img['id'] for img in json_data['images']}
            annotation_image_ids = {ann['image_id'] for ann in json_data['annotations']}
            
            missing_images = annotation_image_ids - image_ids
            if missing_images:
                validation_result['issues'].append(f'标注引用了不存在的图片ID: {list(missing_images)}')
                validation_result['valid'] = False
        
        return jsonify(validation_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualize_json', methods=['POST'])
def visualize_json():
    """可视化JSON文件中的关键点"""
    try:
        data = request.json
        json_path = data.get('json_path')
        image_index = int(data.get('image_index', 0))
        
        if not json_path or not os.path.exists(json_path):
            return jsonify({'error': 'JSON文件不存在'}), 400
        
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if not json_data.get('images') or image_index >= len(json_data['images']):
            return jsonify({'error': '图片索引超出范围'}), 400
        
        image_info = json_data['images'][image_index]
        image_id = image_info['id']
        
        # 找到对应的标注
        annotations = [ann for ann in json_data['annotations'] if ann['image_id'] == image_id]
        
        # 提取关键点信息
        keypoints_data = []
        for ann in annotations:
            if 'keypoints' in ann:
                keypoints = ann['keypoints']
                # 每3个值为一组 (x, y, visibility)
                for i in range(0, len(keypoints), 3):
                    if i + 2 < len(keypoints):
                        keypoints_data.append({
                            'x': keypoints[i],
                            'y': keypoints[i + 1],
                            'visibility': keypoints[i + 2],
                            'point_id': i // 3
                        })
        
        result = {
            'image_info': image_info,
            'keypoints': keypoints_data,
            'annotations_count': len(annotations)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def select_best_keypoints(result):
    """从单帧结果中选择一组关键点（优先置信度最高的目标）。返回归一化坐标列表"""
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
    
    # 转换为列表格式 [{x, y, v}, ...]
    kp_list = []
    for i in range(min(17, arr.shape[0])):
        kp_list.append({
            'x': float(arr[i, 0]),
            'y': float(arr[i, 1]),
            'v': 2  # visibility: 2=可见
        })
    return kp_list

@app.route('/api/get_file')
def get_file():
    """获取文件内容（用于JSON或视频文件）"""
    try:
        file_path = request.args.get('path')
        if not file_path:
            return jsonify({'error': '文件路径未指定'}), 400
        
        if not os.path.exists(file_path):
            return jsonify({'error': '文件不存在'}), 404
        
        # 如果是JSON文件，返回JSON
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        # 如果是视频文件，返回文件
        elif file_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return send_file(file_path, mimetype='video/mp4')
        else:
            return jsonify({'error': '不支持的文件类型'}), 400
            
    except Exception as e:
        logging.error(f'获取文件失败: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_video_first_frame', methods=['POST'])
def get_video_first_frame():
    """获取视频第一帧，用于ROI选择"""
    try:
        data = request.json or {}
        video_path = data.get('video_path')
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': '视频文件不存在'}), 400
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'error': '无法打开视频'}), 400
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'error': '无法读取视频帧'}), 400
        
        # 编码为base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        h, w = frame.shape[:2]
        return jsonify({
            'frame': f'data:image/jpeg;base64,{frame_base64}',
            'width': int(w),
            'height': int(h)
        })
    except Exception as e:
        logging.exception('获取视频第一帧失败')
        return jsonify({'error': str(e)}), 500

@app.route('/api/start_video_pose', methods=['POST'])
def start_video_pose():
    """启动视频关键点检测并通过SSE推送结果"""
    global pose_stream_state
    
    if YOLO is None:
        return jsonify({'error': 'ultralytics未安装，请先安装: pip install ultralytics'}), 500
    
    if pose_stream_state['running']:
        return jsonify({'error': '视频关键点流已在运行中'}), 400
    
    try:
        data = request.json or {}
        video_path = data.get('video_path')
        model_path = data.get('model_path', './models/yolov8n-pose.pt')
        conf_threshold = float(data.get('conf_threshold', 0.5))
        roi_str = data.get('roi')  # 格式: "x,y,w,h" 或 None
        playback_speed = float(data.get('playback_speed', 1.0))  # 播放速度倍数，1.0=正常速度
        action_model_path = data.get('action_model_path')  # 动作识别模型路径（可选）
        action_model_type = data.get('action_model_type', 'stgcn')  # 模型类型：lstm, gru, transformer, stgcn
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': '视频文件不存在'}), 400
        if not os.path.exists(model_path):
            return jsonify({'error': '模型文件不存在'}), 400
        
        # 解析ROI
        roi = None
        if roi_str:
            try:
                parts = roi_str.split(',')
                if len(parts) == 4:
                    roi = tuple(int(p.strip()) for p in parts)
            except Exception as e:
                logging.warning(f'解析ROI失败: {e}，将使用全画面')
        
        q = queue.Queue(maxsize=32)
        pose_stream_state['queue'] = q
        pose_stream_state['stop_flag'] = False
        
        def worker():
            cap = None
            action_model = None
            keypoint_buffer = None
            action_names = None
            sequence_length = 30
            
            try:
                logging.info('加载YOLO姿态模型中...')
                model = YOLO(model_path)
                pose_stream_state['model'] = model
                logging.info('模型加载完成')
                
                # 加载动作识别模型（如果提供）
                if action_model_path and os.path.exists(action_model_path):
                    try:
                        import torch
                        from collections import deque
                        from action_classifier import (
                            ActionClassifier, ActionClassifierGRU, 
                            ActionClassifierTransformer, ActionClassifierSTGCN
                        )
                        
                        logging.info(f'加载动作识别模型 ({action_model_type}): {action_model_path}')
                        checkpoint = torch.load(action_model_path, map_location='cpu')
                        
                        # 从checkpoint获取模型类型（如果存在）
                        checkpoint_model_type = checkpoint.get('model_type', action_model_type)
                        num_classes = checkpoint.get('num_classes', 6)
                        action_map = checkpoint.get('action_map', {})
                        sequence_length = checkpoint.get('sequence_length', 30)
                        
                        # 根据模型类型创建模型
                        if checkpoint_model_type == 'stgcn':
                            num_nodes = checkpoint.get('num_nodes', 17)
                            in_channels = checkpoint.get('in_channels', 2)
                            action_model = ActionClassifierSTGCN(
                                num_nodes=num_nodes,
                                in_channels=in_channels,
                                num_classes=num_classes
                            )
                        elif checkpoint_model_type == 'lstm':
                            input_dim = checkpoint.get('input_dim', 34)
                            hidden_dim = checkpoint.get('hidden_dim', 128)
                            num_layers = checkpoint.get('num_layers', 2)
                            action_model = ActionClassifier(
                                input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                num_layers=num_layers,
                                num_classes=num_classes
                            )
                        elif checkpoint_model_type == 'gru':
                            input_dim = checkpoint.get('input_dim', 34)
                            hidden_dim = checkpoint.get('hidden_dim', 128)
                            num_layers = checkpoint.get('num_layers', 2)
                            action_model = ActionClassifierGRU(
                                input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                num_layers=num_layers,
                                num_classes=num_classes
                            )
                        elif checkpoint_model_type == 'transformer':
                            input_dim = checkpoint.get('input_dim', 34)
                            d_model = checkpoint.get('hidden_dim', 128)
                            action_model = ActionClassifierTransformer(
                                input_dim=input_dim,
                                d_model=d_model,
                                num_classes=num_classes
                            )
                        else:
                            raise ValueError(f"不支持的模型类型: {checkpoint_model_type}")
                        
                        action_model.load_state_dict(checkpoint['model_state_dict'])
                        action_model.eval()
                        
                        # 创建动作名称映射
                        action_names = {v: k for k, v in action_map.items()}
                        if not action_names:
                            action_names = {0: 'W-前进', 1: 'A-左', 2: 'S-后退', 3: 'D-右', 4: '空格-跳跃', 5: 'I-静止'}
                        
                        # 初始化关键点缓冲区
                        keypoint_buffer = deque(maxlen=sequence_length)
                        
                        pose_stream_state['action_model'] = action_model
                        pose_stream_state['action_model_type'] = checkpoint_model_type
                        logging.info(f'{checkpoint_model_type.upper()}模型加载完成，序列长度: {sequence_length}')
                    except Exception as e:
                        logging.warning(f'加载动作识别模型失败: {e}，将仅进行关键点检测')
                        action_model = None
                        keypoint_buffer = None
                else:
                    logging.info('未提供动作识别模型路径，将仅进行关键点检测')
                    action_model = None
                    keypoint_buffer = None
                
                logging.info('开始读取视频')
                
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    q.put({'type': 'error', 'message': '无法打开视频'})
                    return
                
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                q.put({'type': 'meta', 'fps': fps, 'width': width, 'height': height})
                
                # 计算每帧间隔时间（秒），考虑播放速度
                base_interval = 1.0 / fps if fps > 0 else 0.04  # 默认25fps
                frame_interval = base_interval / playback_speed  # 速度越快，间隔越短
                
                center_ema = np.array([0.5, 0.5], dtype=np.float64)
                alpha = 0.2
                frame_idx = 0
                
                while not pose_stream_state['stop_flag']:
                    frame_start_time = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    h_full, w_full = frame.shape[:2]
                    frame_idx += 1
                    
                    # 在ROI区域内推理
                    if roi is not None:
                        x, y, w_roi, h_roi = roi
                        if x >= 0 and y >= 0 and x + w_roi <= w_full and y + h_roi <= h_full:
                            roi_frame = frame[y:y+h_roi, x:x+w_roi]
                            if roi_frame.size > 0:
                                results = model(roi_frame, conf=conf_threshold)
                            else:
                                results = []
                        else:
                            results = []
                    else:
                        results = model(frame, conf=conf_threshold)
                    
                    # 提取关键点
                    best_kp = None
                    for r in results:
                        kp_local = select_best_keypoints(r)
                        if kp_local is not None:
                            # 如果使用了ROI，转换坐标
                            if roi is not None:
                                x, y, w_roi, h_roi = roi
                                for kp in kp_local:
                                    kp['x'] = (kp['x'] * w_roi + x) / w_full
                                    kp['y'] = (kp['y'] * h_roi + y) / h_full
                            best_kp = kp_local
                            break
                    
                    # EMA平滑中心（用于居中）
                    if best_kp:
                        kp_arr = np.array([[p['x'], p['y']] for p in best_kp])
                        x_min = float(kp_arr[:, 0].min())
                        x_max = float(kp_arr[:, 0].max())
                        y_min = float(kp_arr[:, 1].min())
                        y_max = float(kp_arr[:, 1].max())
                        bbox_center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
                        center_ema = (1.0 - alpha) * center_ema + alpha * bbox_center
                    
                    # 动作识别
                    current_action = None
                    action_confidence = 0.0
                    model_type = pose_stream_state.get('action_model_type', 'stgcn')
                    
                    if action_model is not None and best_kp:
                        try:
                            import torch
                            # 将关键点转换为numpy数组 (17, 2)
                            kp_array = np.array([[p['x'], p['y']] for p in best_kp], dtype=np.float32)
                            # 添加到缓冲区
                            keypoint_buffer.append(kp_array.flatten())  # (34,)
                            
                            # 当缓冲区满时进行推理（每次缓冲区满时都推理，实现实时更新）
                            if len(keypoint_buffer) >= sequence_length:
                                seq_array = np.array(list(keypoint_buffer), dtype=np.float32)  # (seq_len, 34)
                                
                                # 根据模型类型准备不同的输入格式
                                if model_type == 'stgcn':
                                    # ST-GCN需要 (1, seq_len, 17, 2) 格式
                                    seq_tensor = torch.FloatTensor(seq_array.reshape(1, sequence_length, 17, 2))
                                else:
                                    # LSTM/GRU/Transformer需要 (1, seq_len, 34) 格式
                                    seq_tensor = torch.FloatTensor(seq_array.reshape(1, sequence_length, 34))
                                
                                # 推理
                                with torch.no_grad():
                                    outputs = action_model(seq_tensor)
                                    probs = torch.softmax(outputs, dim=1)
                                    confidence, predicted = torch.max(probs, 1)
                                    
                                    action_id = predicted.item()
                                    action_confidence = confidence.item()
                                    current_action = action_names.get(action_id, f"class_{action_id}")
                        except Exception as e:
                            logging.warning(f'{model_type.upper()}推理失败: {e}')
                            current_action = None
                    elif action_model is not None and not best_kp and keypoint_buffer:
                        # 如果没有检测到关键点，使用上一帧填充
                        if len(keypoint_buffer) > 0:
                            keypoint_buffer.append(keypoint_buffer[-1])
                            # 如果缓冲区已满，也进行推理（使用重复的关键点）
                            if len(keypoint_buffer) >= sequence_length:
                                try:
                                    import torch
                                    seq_array = np.array(list(keypoint_buffer), dtype=np.float32)
                                    
                                    if model_type == 'stgcn':
                                        seq_tensor = torch.FloatTensor(seq_array.reshape(1, sequence_length, 17, 2))
                                    else:
                                        seq_tensor = torch.FloatTensor(seq_array.reshape(1, sequence_length, 34))
                                    
                                    with torch.no_grad():
                                        outputs = action_model(seq_tensor)
                                        probs = torch.softmax(outputs, dim=1)
                                        confidence, predicted = torch.max(probs, 1)
                                        
                                        action_id = predicted.item()
                                        action_confidence = confidence.item()
                                        current_action = action_names.get(action_id, f"class_{action_id}")
                                except Exception as e:
                                    logging.warning(f'{model_type.upper()}推理失败: {e}')
                                    current_action = None
                    
                    # 每帧都发送视频帧，通过降低质量和适度缩放来平衡性能
                    frame_base64 = None
                    # 在帧上绘制关键点和ROI框
                    display_frame = frame.copy()
                    
                    # 如果帧太大，先缩小以提升编码速度
                    max_display_width = 960
                    scale_factor = 1.0
                    if w_full > max_display_width:
                        scale_factor = max_display_width / w_full
                        new_w = int(w_full * scale_factor)
                        new_h = int(h_full * scale_factor)
                        display_frame = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    
                    if roi is not None:
                        x_roi, y_roi, w_roi, h_roi = roi
                        x_roi_scaled = int(x_roi * scale_factor)
                        y_roi_scaled = int(y_roi * scale_factor)
                        w_roi_scaled = int(w_roi * scale_factor)
                        h_roi_scaled = int(h_roi * scale_factor)
                        cv2.rectangle(display_frame, (x_roi_scaled, y_roi_scaled), 
                                     (x_roi_scaled + w_roi_scaled, y_roi_scaled + h_roi_scaled), (255, 0, 0), 2)
                    
                    if best_kp:
                        h, w = display_frame.shape[:2]
                        pts_px = []
                        for p in best_kp:
                            pts_px.append((int(p['x'] * w), int(p['y'] * h)))
                        
                        # 画点和线
                        for (x, y) in pts_px:
                            cv2.circle(display_frame, (x, y), 3, (0, 255, 255), -1)
                        
                        COCO_EDGES = [
                            [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6],
                            [5, 7], [7, 9], [6, 8], [8, 10],
                            [11, 13], [13, 15], [12, 14], [14, 16],
                            [5, 6], [11, 12], [5, 11], [6, 12]
                        ]
                        for e in COCO_EDGES:
                            if e[0] < len(pts_px) and e[1] < len(pts_px):
                                cv2.line(display_frame, pts_px[e[0]], pts_px[e[1]], (0, 200, 255), 2)
                    
                    # 压缩编码：降低质量以平衡性能和流畅度（60-75质量是好的平衡点）
                    _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    payload = {
                        'type': 'keypoints',
                        'people': [best_kp] if best_kp else [],
                        'center': [float(center_ema[0]), float(center_ema[1])],
                        'frame': frame_base64,  # 每帧都包含视频帧数据
                        'ts': time.time()
                    }
                    
                    # 添加动作识别结果
                    # 每次帧都发送动作识别结果（如果缓冲区已满）
                    if action_model is not None and keypoint_buffer and len(keypoint_buffer) >= sequence_length:
                        if current_action is not None:
                            payload['action'] = current_action
                            payload['action_confidence'] = float(action_confidence)
                        # 即使current_action为None，也发送一个标记，表示缓冲区已满但当前帧没有结果
                        # 这样前端可以知道系统正在运行
                    
                    try:
                        q.put(payload, timeout=0.5)
                    except queue.Full:
                        # 丢弃旧帧，保持实时
                        try:
                            _ = q.get_nowait()
                        except Exception:
                            pass
                        try:
                            q.put(payload, timeout=0.1)
                        except Exception:
                            pass
                    
                    # 帧率控制：确保按照视频原始FPS播放
                    frame_end_time = time.time()
                    elapsed = frame_end_time - frame_start_time
                    sleep_time = frame_interval - elapsed
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)  # 等待剩余时间
                    # 如果处理时间超过帧间隔，立即处理下一帧（保持实时性）
                    
            except Exception as e:
                logging.exception('视频关键点工作线程异常')
                try:
                    q.put({'type': 'error', 'message': str(e)})
                except Exception:
                    pass
            finally:
                if cap is not None:
                    cap.release()
                pose_stream_state['running'] = False
                pose_stream_state['model'] = None
                pose_stream_state['action_model'] = None
                try:
                    q.put({'type': 'eof'})
                except Exception:
                    pass
        
        t = threading.Thread(target=worker, daemon=True)
        pose_stream_state['thread'] = t
        pose_stream_state['running'] = True
        t.start()
        return jsonify({'message': '视频关键点流已启动'})
        
    except Exception as e:
        pose_stream_state['running'] = False
        pose_stream_state['stop_flag'] = False
        pose_stream_state['thread'] = None
        pose_stream_state['queue'] = None
        logging.exception('启动视频关键点流失败')
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_video_pose', methods=['POST'])
def stop_video_pose():
    """停止视频关键点流"""
    global pose_stream_state
    if not pose_stream_state['running']:
        return jsonify({'message': '未在运行'})
    pose_stream_state['stop_flag'] = True
    return jsonify({'message': '停止指令已发送'})

@app.route('/api/pose_stream')
def pose_stream():
    """SSE推送关键点: text/event-stream，每条为JSON"""
    global pose_stream_state
    q = pose_stream_state.get('queue')
    if q is None:
        return jsonify({'error': '流未启动'}), 400
    
    @stream_with_context
    def event_source():
        last_heartbeat = time.time()
        while True:
            # 心跳，防止中间件/浏览器超时断开
            now = time.time()
            if now - last_heartbeat >= 10:
                yield ': heartbeat\n\n'
                last_heartbeat = now
            
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                if not pose_stream_state['running']:
                    yield 'event: end\ndata: {}\n\n'
                    break
                continue
            
            if item is None:
                continue
            
            if item.get('type') == 'eof':
                yield 'event: end\ndata: {}\n\n'
                break
            
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
    
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'text/event-stream',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no'
    }
    return Response(event_source(), headers=headers, mimetype='text/event-stream')

@app.route('/api/start_video_action_labeling', methods=['POST'])
def start_video_action_labeling():
    """启动视频动作标注"""
    global action_labeling_state
    
    if YOLO is None:
        return jsonify({'error': 'ultralytics未安装'}), 500
    
    if action_labeling_state['video_labeling']['running']:
        return jsonify({'error': '视频标注已在运行中'}), 400
    
    try:
        data = request.json or {}
        video_path = data.get('video_path')
        model_path = data.get('model_path', './models/yolov8n-pose.pt')
        output_path = data.get('output_path')
        conf_threshold = float(data.get('conf_threshold', 0.5))
        seq_len = int(data.get('seq_len', 30))
        roi_str = data.get('roi')
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': '视频文件不存在'}), 400
        if not os.path.exists(model_path):
            return jsonify({'error': '模型文件不存在'}), 400
        
        # 确定输出路径
        if not output_path:
            video_name = Path(video_path).stem
            output_path = f"{video_name}_action_labels.json"
        
        # 解析ROI
        roi = None
        if roi_str:
            try:
                parts = roi_str.split(',')
                if len(parts) == 4:
                    roi = tuple(int(p.strip()) for p in parts)
            except Exception:
                pass
        
        # 导入标注函数
        from video_action_labeler import extract_keypoints_from_video, ACTION_MAP
        
        def worker():
            try:
                action_labeling_state['video_labeling']['running'] = True
                logging.info(f'开始视频动作标注: {video_path}')
                
                # 提取关键点
                keypoint_sequence, frame_indices, fps = extract_keypoints_from_video(
                    video_path, model_path, conf_threshold, roi
                )
                
                # 创建空的标注数据（用户需要在客户端进行标注）
                data = {
                    'video_path': video_path,
                    'total_frames': len(keypoint_sequence),
                    'fps': fps,
                    'keypoint_sequence': keypoint_sequence,
                    'frame_indices': frame_indices,
                    'annotations': {},
                    'action_map': ACTION_MAP
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                action_labeling_state['video_labeling']['output_path'] = output_path
                logging.info(f'关键点提取完成，保存到: {output_path}')
                
            except Exception as e:
                logging.error(f'视频标注错误: {e}', exc_info=True)
            finally:
                action_labeling_state['video_labeling']['running'] = False
        
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        action_labeling_state['video_labeling']['thread'] = t
        
        return jsonify({
            'message': '视频标注已启动',
            'output_path': output_path
        })
        
    except Exception as e:
        logging.error(f'启动视频标注失败: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_video_labeling_status')
def get_video_labeling_status():
    """获取视频标注状态"""
    global action_labeling_state
    state = action_labeling_state['video_labeling']
    return jsonify({
        'running': state['running'],
        'output_path': state.get('output_path')
    })

@app.route('/api/save_video_annotation', methods=['POST'])
def save_video_annotation():
    """保存视频标注"""
    try:
        data = request.json or {}
        output_path = data.get('output_path')
        annotations = data.get('annotations', {})
        
        if not output_path:
            return jsonify({'error': '输出路径未指定'}), 400
        
        # 加载现有数据
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
        else:
            return jsonify({'error': '标注文件不存在'}), 400
        
        # 更新标注
        file_data['annotations'] = annotations
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(file_data, f, indent=2, ensure_ascii=False)
        
        return jsonify({'message': '标注已保存'})
        
    except Exception as e:
        logging.error(f'保存标注失败: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/start_realtime_labeling', methods=['POST'])
def start_realtime_labeling():
    """启动实时游戏标注"""
    global action_labeling_state
    
    if YOLO is None:
        return jsonify({'error': 'ultralytics未安装'}), 500
    
    if action_labeling_state['realtime_labeling']['running']:
        return jsonify({'error': '实时标注已在运行中'}), 400
    
    try:
        data = request.json or {}
        model_path = data.get('model_path', './models/yolov8n-pose.pt')
        monitor_str = data.get('monitor')
        conf_threshold = float(data.get('conf_threshold', 0.5))
        fps = float(data.get('fps', 30))
        output_path = data.get('output_path')
        seq_len = int(data.get('seq_len', 30))
        
        if not os.path.exists(model_path):
            return jsonify({'error': '模型文件不存在'}), 400
        
        # 确定输出路径
        if not output_path:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"game_action_labels_{timestamp}.json"
        
        # 解析monitor
        monitor = None
        if monitor_str:
            try:
                parts = monitor_str.split(',')
                if len(parts) == 4:
                    monitor = tuple(int(p.strip()) for p in parts)
            except Exception:
                pass
        
        # 检查依赖
        try:
            from pynput import keyboard
        except ImportError:
            return jsonify({'error': '需要安装pynput: pip install pynput'}), 500
        
        try:
            import mss
        except ImportError:
            logging.warning('mss未安装，将使用PIL进行屏幕捕获')
        
        # 导入实时标注类
        from realtime_game_labeler import RealtimeGameLabeler
        
        def worker():
            try:
                action_labeling_state['realtime_labeling']['running'] = True
                action_labeling_state['realtime_labeling']['stop_flag'] = False
                
                labeler = RealtimeGameLabeler(
                    model_path=model_path,
                    monitor=monitor,
                    conf_threshold=conf_threshold,
                    sequence_length=seq_len,
                    output_path=output_path,
                    fps=fps
                )
                
                # 设置停止标志检查
                def check_stop():
                    while action_labeling_state['realtime_labeling']['running']:
                        if action_labeling_state['realtime_labeling'].get('stop_flag'):
                            labeler.stop_flag = True
                            break
                        time.sleep(0.1)
                
                stop_thread = threading.Thread(target=check_stop, daemon=True)
                stop_thread.start()
                
                try:
                    labeler.run()
                except Exception as e:
                    logging.error(f'实时标注运行错误: {e}', exc_info=True)
                
            except Exception as e:
                logging.error(f'实时标注错误: {e}', exc_info=True)
            finally:
                action_labeling_state['realtime_labeling']['running'] = False
                action_labeling_state['realtime_labeling']['output_path'] = output_path
        
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        action_labeling_state['realtime_labeling']['thread'] = t
        
        return jsonify({
            'message': '实时标注已启动',
            'output_path': output_path
        })
        
    except Exception as e:
        logging.error(f'启动实时标注失败: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_realtime_labeling', methods=['POST'])
def stop_realtime_labeling():
    """停止实时标注"""
    global action_labeling_state
    
    if not action_labeling_state['realtime_labeling']['running']:
        return jsonify({'message': '未在运行'})
    
    action_labeling_state['realtime_labeling']['stop_flag'] = True
    return jsonify({'message': '停止指令已发送'})

@app.route('/api/get_realtime_labeling_status')
def get_realtime_labeling_status():
    """获取实时标注状态"""
    global action_labeling_state
    state = action_labeling_state['realtime_labeling']
    return jsonify({
        'running': state['running'],
        'output_path': state.get('output_path')
    })

# 动作训练状态
action_training_state = {
    'running': False,
    'thread': None,
    'stop_flag': False,
    'logs': [],
    'current_epoch': 0,
    'total_epochs': 0,
    'output_dir': './models/action_classifier',
    'model_type': 'lstm'
}

@app.route('/api/start_action_training', methods=['POST'])
def start_action_training():
    """启动动作分类模型训练"""
    global action_training_state
    
    if action_training_state['running']:
        return jsonify({'error': '训练已在运行中'}), 400
    
    try:
        data = request.json or {}
        data_path = data.get('data_path')
        # 默认使用ST-GCN作为训练模型类型
        model_type = data.get('model_type', 'stgcn')
        seq_len = int(data.get('seq_len', 30))
        hidden_dim = int(data.get('hidden_dim', 128))
        num_layers = int(data.get('num_layers', 2))
        batch_size = int(data.get('batch_size', 32))
        epochs = int(data.get('epochs', 50))
        learning_rate = float(data.get('learning_rate', 0.001))
        output_dir = data.get('output_dir', './models/action_classifier')
        
        if not data_path:
            return jsonify({'error': '请指定标注数据路径'}), 400
        
        if not os.path.exists(data_path):
            return jsonify({'error': '数据路径不存在'}), 400
        
        # 导入训练函数（在worker外部导入，避免在worker内部导入时输出被捕获）
        from train_action_classifier import train_action_classifier
        
        def worker():
            # 初始化状态
            action_training_state['running'] = True
            action_training_state['stop_flag'] = False
            action_training_state['current_epoch'] = 0
            action_training_state['total_epochs'] = epochs
            action_training_state['output_dir'] = output_dir
            action_training_state['model_type'] = model_type
            
            # 创建日志文件路径
            log_file_path = os.path.join(output_dir, f'training_log_{model_type}.txt')
            os.makedirs(output_dir, exist_ok=True)
            # 立即保存日志文件路径到状态，确保前端能获取到
            action_training_state['log_file'] = log_file_path
            
            try:
                # 重定向stdout到文件
                import sys
                original_stdout = sys.stdout
                
                # 打开日志文件（写入模式）
                log_file = open(log_file_path, 'w', encoding='utf-8')
                
                # 创建一个同时写入文件和原始stdout的类
                class TeeOutput:
                    def __init__(self, file, original):
                        self.file = file
                        self.original = original
                    
                    def write(self, text):
                        self.file.write(text)
                        self.file.flush()
                        self.original.write(text)
                        self.original.flush()
                    
                    def flush(self):
                        self.file.flush()
                        self.original.flush()
                    
                    def __getattr__(self, name):
                        return getattr(self.original, name)
                
                sys.stdout = TeeOutput(log_file, original_stdout)
                
                try:
                    print("=" * 60)
                    print("开始训练动作分类模型")
                    print(f"模型类型: {model_type}")
                    print(f"训练轮数: {epochs}")
                    print(f"数据路径: {data_path}")
                    print(f"输出目录: {output_dir}")
                    print("=" * 60)
                    
                    train_action_classifier(
                        data_dir=data_path,
                        model_type=model_type,
                        sequence_length=seq_len,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        batch_size=batch_size,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        device='cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu',
                        output_dir=output_dir
                    )
                    
                    print("=" * 60)
                    print("训练完成！")
                    print("=" * 60)
                    
                except Exception as e:
                    import traceback
                    print("=" * 60)
                    print(f"训练错误: {str(e)}")
                    print("=" * 60)
                    print("错误详情:")
                    traceback.print_exc()
                    logging.error(f'训练错误: {e}', exc_info=True)
                finally:
                    # 恢复原始stdout并关闭文件
                    sys.stdout = original_stdout
                    log_file.close()
                    # 确保日志文件路径已保存
                    action_training_state['log_file'] = log_file_path
                    logging.info(f'训练日志已保存到: {log_file_path}')
                    action_training_state['running'] = False
                    
            except Exception as e:
                logging.error(f'训练线程错误: {e}', exc_info=True)
                action_training_state['running'] = False
        
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        action_training_state['thread'] = t
        
        # 等待一小段时间，确保worker函数开始执行并初始化日志
        import time
        time.sleep(0.1)
        
        # 检查日志是否已初始化
        logs_count = len(action_training_state.get('logs', []))
        logging.info(f'训练线程已启动，当前日志数: {logs_count}')
        
        return jsonify({
            'message': '训练已启动',
            'initial_logs_count': logs_count
        })
        
    except Exception as e:
        logging.error(f'启动训练失败: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_action_training', methods=['POST'])
def stop_action_training():
    """停止训练"""
    global action_training_state
    
    if not action_training_state['running']:
        return jsonify({'message': '未在运行'})
    
    action_training_state['stop_flag'] = True
    return jsonify({'message': '停止指令已发送'})

@app.route('/api/get_training_status')
def get_training_status():
    """获取训练状态"""
    global action_training_state
    output_dir = action_training_state.get('output_dir', './models/action_classifier')
    model_type = action_training_state.get('model_type', 'lstm')
    
    # 检查训练结果图片是否存在
    training_curves_path = os.path.join(output_dir, f'training_curves_{model_type}.png')
    confusion_matrix_path = os.path.join(output_dir, f'confusion_matrix_{model_type}.png')
    log_file_path = action_training_state.get('log_file', os.path.join(output_dir, f'training_log_{model_type}.txt'))
    
    result = {
        'running': action_training_state.get('running', False),
        'current_epoch': action_training_state.get('current_epoch', 0),
        'total_epochs': action_training_state.get('total_epochs', 0),
        'has_training_curves': os.path.exists(training_curves_path),
        'has_confusion_matrix': os.path.exists(confusion_matrix_path),
        'has_log_file': os.path.exists(log_file_path),
        'log_file_path': log_file_path if os.path.exists(log_file_path) else None
    }
    
    if result['has_training_curves']:
        result['training_curves_path'] = training_curves_path
    if result['has_confusion_matrix']:
        result['confusion_matrix_path'] = confusion_matrix_path
    
    return jsonify(result)

@app.route('/api/get_training_log')
def get_training_log():
    """获取训练日志文件内容"""
    global action_training_state
    output_dir = action_training_state.get('output_dir', './models/action_classifier')
    model_type = action_training_state.get('model_type', 'lstm')
    log_file_path = action_training_state.get('log_file', os.path.join(output_dir, f'training_log_{model_type}.txt'))
    
    if not os.path.exists(log_file_path):
        return jsonify({'error': '日志文件不存在', 'log_file_path': log_file_path}), 404
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        return jsonify({
            'log_content': log_content,
            'log_file_path': log_file_path
        })
    except Exception as e:
        logging.error(f'读取日志文件失败: {e}', exc_info=True)
        return jsonify({'error': f'读取日志文件失败: {str(e)}'}), 500

@app.route('/api/get_training_image')
def get_training_image():
    """获取训练结果图片"""
    try:
        image_path = request.args.get('path')
        if not image_path:
            return jsonify({'error': '图片路径未指定'}), 400
        
        # 转换为绝对路径
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.getcwd(), image_path.lstrip('./'))
        
        # 安全检查：确保路径在允许的目录内
        if not os.path.exists(image_path):
            return jsonify({'error': '图片文件不存在'}), 404
        
        # 只允许访问模型输出目录下的图片
        allowed_base = os.path.join(os.getcwd(), 'models', 'action_classifier')
        image_abs = os.path.abspath(image_path)
        if not image_abs.startswith(os.path.abspath(allowed_base)):
            return jsonify({'error': '不允许访问该路径'}), 403
        
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        logging.error(f'获取训练图片失败: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    setup_logging()
    
    # 创建templates目录和HTML文件
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # 只在文件不存在时生成，避免覆盖用户修改
    html_file = templates_dir / 'index.html'
    if not html_file.exists():
        logging.info('index.html不存在，正在生成...')
        # 创建HTML模板
        html_content = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MotionSync - 位姿提取与动作估计软件</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .content {
            padding: 20px;
        }
        .tab-container {
            display: flex;
            border-bottom: 2px solid #eee;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.3s;
        }
        .tab.active {
            border-bottom-color: #667eea;
            color: #667eea;
            font-weight: bold;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        .btn {
            background: #667eea;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 10px;
        }
        .btn:hover {
            background: #5a6fd8;
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .progress {
            width: 100%;
            height: 20px;
            background: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-bar {
            height: 100%;
            background: #667eea;
            transition: width 0.3s;
        }
        .log-container {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 15px;
            height: 300px;
            overflow-y: auto;
            font-family: 'Consolas', monospace;
            font-size: 12px;
        }
        .status {
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .status.info {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        .dataset-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .info-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }
        .info-card h3 {
            margin: 0 0 10px 0;
            color: #667eea;
        }
        .info-card p {
            margin: 0;
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 MotionSync</h1>
            <p>位姿提取与动作估计软件 - Web版本</p>
        </div>
        
        <div class="content">
            <div class="tab-container">
                <div class="tab active" onclick="showTab('auto-label')">位姿提取</div>
                <div class="tab" onclick="showTab('convert')">格式转换</div>
                <div class="tab" onclick="showTab('validate')">JSON验证</div>
                <div class="tab" onclick="showTab('coco')">COCO Annotator</div>
                <div class="tab" onclick="showTab('dataset')">数据集信息</div>
                <div class="tab" onclick="showTab('logs')">运行日志</div>
                <div class="tab" onclick="showTab('pose3d')">3D火柴人</div>
                <div class="tab" onclick="showTab('video_action')">视频动作标注</div>
                <div class="tab" onclick="showTab('realtime_action')">实时游戏标注</div>
            </div>
            
            <!-- 位姿提取标签页 -->
            <div id="auto-label" class="tab-content active">
                <h2>位姿提取</h2>
                <div class="form-group">
                    <label>图片目录:</label>
                    <input type="text" id="img_dir" value="./datasets/images/train">
                </div>
                <div class="form-group">
                    <label>模型路径:</label>
                    <input type="text" id="model_path" value="./models/yolov8n-pose.pt">
                </div>
                <div class="form-group">
                    <label>输出目录:</label>
                    <input type="text" id="output_dir" value="./datasets/labels/train">
                </div>
                <div class="form-group">
                    <label>置信度阈值:</label>
                    <input type="number" id="conf_threshold" value="0.5" min="0.1" max="1.0" step="0.1">
                </div>
                <button class="btn" onclick="startAutoLabel()">开始位姿提取</button>
                <div class="progress" id="progress" style="display: none;">
                    <div class="progress-bar" id="progress-bar"></div>
                </div>
                <div id="status"></div>
            </div>
            
            <!-- 格式转换标签页 -->
            <div id="convert" class="tab-content">
                <h2>格式转换</h2>
                <div class="form-group">
                    <label>转换类型:</label>
                    <select id="convert_mode">
                        <option value="txt2coco">TXT → COCO JSON</option>
                        <option value="coco2txt">COCO JSON → TXT</option>
                        <option value="txt2labelme">TXT → LabelMe JSON</option>
                        <option value="labelme2txt">LabelMe JSON → TXT</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>输入路径:</label>
                    <input type="text" id="input_path" value="./datasets/labels/train">
                </div>
                <div class="form-group">
                    <label>输出路径:</label>
                    <input type="text" id="output_path" value="./datasets/labels_json/1.json">
                </div>
                <div class="form-group">
                    <label>图片目录:</label>
                    <input type="text" id="convert_img_dir" value="./datasets/images/train">
                </div>
                <button class="btn" onclick="startConvert()">开始转换</button>
                <div class="progress" id="convert-progress" style="display: none;">
                    <div class="progress-bar" id="convert-progress-bar"></div>
                </div>
                <div id="convert-status"></div>
            </div>
            
            <!-- JSON验证标签页 -->
            <div id="validate" class="tab-content">
                <h2>JSON验证与可视化</h2>
                <div class="form-group">
                    <label>JSON文件路径:</label>
                    <input type="text" id="json_path" placeholder="输入JSON文件路径">
                </div>
                <button class="btn" onclick="validateJson()">验证JSON</button>
                <button class="btn" onclick="visualizeJson()">可视化关键点</button>
                
                <div id="validation-result" style="margin-top: 20px;"></div>
                
                <div id="visualization-section" style="margin-top: 20px; display: none;">
                    <h3>关键点可视化</h3>
                    <div class="form-group">
                        <label>图片索引:</label>
                        <input type="number" id="image_index" value="0" min="0">
                    </div>
                    <div id="keypoints-display"></div>
                </div>
            </div>
            
            <!-- COCO Annotator标签页 -->
            <div id="coco" class="tab-content">
                <h2>COCO Annotator管理</h2>
                <div class="info-card">
                    <h3>COCO Annotator状态</h3>
                    <p id="coco-status">检查中...</p>
                </div>
                
                <div style="margin: 20px 0;">
                    <button class="btn" onclick="checkCocoStatus()">检查状态</button>
                    <button class="btn" onclick="startCocoAnnotator()">启动COCO Annotator</button>
                    <button class="btn" onclick="stopCocoAnnotator()">停止COCO Annotator</button>
                    <button class="btn" onclick="openCocoAnnotator()" id="open-coco-btn" style="display: none;">打开COCO Annotator</button>
                </div>
                
                <div id="coco-message" style="margin-top: 20px;"></div>
                
                <div class="info-card" style="margin-top: 20px;">
                    <h3>使用说明</h3>
                    <p>1. 点击"启动COCO Annotator"启动服务</p>
                    <p>2. 等待启动完成后，点击"打开COCO Annotator"</p>
                    <p>3. 在COCO Annotator中验证和修改JSON文件</p>
                    <p>4. 完成后可以停止服务释放资源</p>
                </div>
            </div>
            
            <!-- 数据集信息标签页 -->
            <div id="dataset" class="tab-content">
                <h2>数据集信息</h2>
                <button class="btn" onclick="loadDatasetInfo()">刷新信息</button>
                <div class="dataset-info" id="dataset-info">
                    <!-- 数据集信息将在这里显示 -->
                </div>
            </div>
            
            <!-- 运行日志标签页 -->
            <div id="logs" class="tab-content">
                <h2>运行日志</h2>
                <button class="btn" onclick="loadLogs()">刷新日志</button>
                <button class="btn" onclick="clearLogs()">清空日志</button>
                <div class="log-container" id="log-container">
                    <!-- 日志将在这里显示 -->
                </div>
            </div>
            
            <!-- 3D火柴人标签页 -->
            <div id="pose3d" class="tab-content">
                <h2>3D火柴人（实时）</h2>
                <div class="form-group">
                    <label>视频路径:</label>
                    <input type="text" id="video_path" placeholder="例如: ./datasets/videos/demo.mp4 或 ./1.mp4">
                    <button class="btn" onclick="loadVideoFrame()" style="margin-top: 5px;">加载第一帧并选择ROI</button>
                </div>
                <div class="form-group">
                    <label>ROI区域:</label>
                    <input type="text" id="pose_roi" placeholder="将在第一帧上选择，或手动输入 x,y,w,h">
                    <div id="roi_canvas_container" style="margin-top: 10px; display: none;">
                        <canvas id="roi_canvas" style="border: 1px solid #ddd; max-width: 100%; cursor: crosshair;"></canvas>
                        <div style="margin-top: 5px;">
                            <button class="btn" onclick="confirmROI()">确认ROI</button>
                            <button class="btn" onclick="cancelROISelection()">取消</button>
                        </div>
                    </div>
                </div>
                <div class="form-group">
                    <label>模型路径:</label>
                    <input type="text" id="pose_model_path" value="./models/yolov8n-pose.pt">
                </div>
                <div class="form-group">
                    <label>置信度阈值:</label>
                    <input type="number" id="pose_conf" value="0.5" min="0.1" max="1.0" step="0.1">
                </div>
                <div class="form-group">
                    <label>缩放比例:</label>
                    <input type="number" id="pose_scale" value="1.0" min="0.1" max="3.0" step="0.1">
                </div>
                <div class="form-group">
                    <label>播放速度:</label>
                    <input type="number" id="pose_speed" value="1.0" min="0.1" max="5.0" step="0.1">
                    <small style="color: #666;">1.0=正常速度，2.0=2倍速，0.5=0.5倍速</small>
                </div>
                <button class="btn" onclick="startPoseStream()">开始</button>
                <button class="btn" onclick="stopPoseStream()">停止</button>

                <div id="pose_status" style="margin-top: 10px;"></div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                    <div>
                        <h3>视频流</h3>
                        <img id="video_frame" style="width: 100%; background: #111; border-radius: 8px; display: none;">
                        <div id="video_placeholder" style="width: 100%; height: 360px; background: #111; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #666;">
                            等待视频流...
                        </div>
                    </div>
                    <div>
                        <h3>3D火柴人</h3>
                        <div id="three-container" style="width: 100%; height: 360px; background: #111; border-radius: 8px;"></div>
                    </div>
                </div>
                
                <div style="margin-top: 10px; color:#666;">
                    提示：点击"加载第一帧并选择ROI"可在视频第一帧上拖拽选择感兴趣区域。3D火柴人实时显示关键点，可拖拽旋转查看不同角度。
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        function showTab(tabName) {
            // 隐藏所有标签页内容
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // 移除所有标签的active类
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // 显示选中的标签页
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }
        
        function startAutoLabel() {
            const data = {
                img_dir: document.getElementById('img_dir').value,
                model_path: document.getElementById('model_path').value,
                output_dir: document.getElementById('output_dir').value,
                conf_threshold: parseFloat(document.getElementById('conf_threshold').value)
            };
            
            fetch('/api/auto_label', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showStatus('error', data.error);
                } else {
                    showStatus('info', data.message);
                    document.getElementById('progress').style.display = 'block';
                    pollStatus();
                }
            })
            .catch(error => {
                showStatus('error', '请求失败: ' + error);
            });
        }
        
        function startConvert() {
            const data = {
                mode: document.getElementById('convert_mode').value,
                input_path: document.getElementById('input_path').value,
                output_path: document.getElementById('output_path').value,
                img_dir: document.getElementById('convert_img_dir').value
            };
            
            fetch('/api/convert', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showConvertStatus('error', data.error);
                } else {
                    showConvertStatus('info', data.message);
                    document.getElementById('convert-progress').style.display = 'block';
                    pollStatus();
                }
            })
            .catch(error => {
                showConvertStatus('error', '请求失败: ' + error);
            });
        }
        
        function pollStatus() {
            fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                if (data.running) {
                    document.getElementById('progress-bar').style.width = data.progress + '%';
                    document.getElementById('convert-progress-bar').style.width = data.progress + '%';
                    setTimeout(pollStatus, 1000);
                } else {
                    document.getElementById('progress').style.display = 'none';
                    document.getElementById('convert-progress').style.display = 'none';
                    if (data.message) {
                        showStatus('success', data.message);
                        showConvertStatus('success', data.message);
                    }
                }
            });
        }
        
        function showStatus(type, message) {
            const statusDiv = document.getElementById('status');
            statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
        }
        
        function showConvertStatus(type, message) {
            const statusDiv = document.getElementById('convert-status');
            statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
        }
        
        function loadDatasetInfo() {
            fetch('/api/dataset_info')
            .then(response => response.json())
            .then(data => {
                const container = document.getElementById('dataset-info');
                container.innerHTML = `
                    <div class="info-card">
                        <h3>训练图片</h3>
                        <p>${data.train_images}</p>
                    </div>
                    <div class="info-card">
                        <h3>验证图片</h3>
                        <p>${data.val_images}</p>
                    </div>
                    <div class="info-card">
                        <h3>训练标签</h3>
                        <p>${data.train_labels}</p>
                    </div>
                    <div class="info-card">
                        <h3>验证标签</h3>
                        <p>${data.val_labels}</p>
                    </div>
                    <div class="info-card">
                        <h3>总图片数</h3>
                        <p>${data.total_images}</p>
                    </div>
                    <div class="info-card">
                        <h3>总标签数</h3>
                        <p>${data.total_labels}</p>
                    </div>
                `;
            });
        }
        
        function loadLogs() {
            fetch('/api/logs')
            .then(response => response.json())
            .then(data => {
                const container = document.getElementById('log-container');
                container.innerHTML = data.logs.map(log => `<div>${log}</div>`).join('');
                container.scrollTop = container.scrollHeight;
            });
        }
        
        function clearLogs() {
            document.getElementById('log-container').innerHTML = '';
        }
        
        function validateJson() {
            const jsonPath = document.getElementById('json_path').value;
            if (!jsonPath) {
                showValidationResult('error', '请输入JSON文件路径');
                return;
            }
            
            fetch('/api/validate_json', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({json_path: jsonPath})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showValidationResult('error', data.error);
                } else {
                    const status = data.valid ? 'success' : 'error';
                    const message = data.valid ? 
                        `JSON文件验证通过！图片: ${data.images_count}, 标注: ${data.annotations_count}, 类别: ${data.categories_count}` :
                        `JSON文件验证失败: ${data.issues.join(', ')}`;
                    showValidationResult(status, message);
                }
            })
            .catch(error => {
                showValidationResult('error', '验证失败: ' + error);
            });
        }
        
        function visualizeJson() {
            const jsonPath = document.getElementById('json_path').value;
            const imageIndex = parseInt(document.getElementById('image_index').value);
            
            if (!jsonPath) {
                showValidationResult('error', '请输入JSON文件路径');
                return;
            }
            
            fetch('/api/visualize_json', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    json_path: jsonPath,
                    image_index: imageIndex
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showValidationResult('error', data.error);
                } else {
                    displayKeypoints(data);
                    document.getElementById('visualization-section').style.display = 'block';
                }
            })
            .catch(error => {
                showValidationResult('error', '可视化失败: ' + error);
            });
        }
        
        function showValidationResult(type, message) {
            const resultDiv = document.getElementById('validation-result');
            resultDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
        }
        
        function displayKeypoints(data) {
            const container = document.getElementById('keypoints-display');
            const imageInfo = data.image_info;
            const keypoints = data.keypoints;
            
            let html = `
                <div class="info-card">
                    <h3>图片信息</h3>
                    <p>文件名: ${imageInfo.file_name}</p>
                    <p>尺寸: ${imageInfo.width} x ${imageInfo.height}</p>
                    <p>标注数量: ${data.annotations_count}</p>
                </div>
                <div class="info-card">
                    <h3>关键点信息</h3>
                    <p>关键点数量: ${keypoints.length}</p>
                </div>
            `;
            
            if (keypoints.length > 0) {
                html += '<h4>关键点详情:</h4><div style="max-height: 300px; overflow-y: auto;">';
                keypoints.forEach((kp, index) => {
                    const visibility = kp.visibility === 2 ? '可见' : kp.visibility === 1 ? '遮挡' : '不可见';
                    html += `
                        <div style="padding: 5px; border-bottom: 1px solid #eee;">
                            关键点 ${kp.point_id}: (${kp.x.toFixed(2)}, ${kp.y.toFixed(2)}) - ${visibility}
                        </div>
                    `;
                });
                html += '</div>';
            }
            
            container.innerHTML = html;
        }
        
        function checkCocoStatus() {
            fetch('/api/coco_annotator_status')
            .then(response => response.json())
            .then(data => {
                const statusElement = document.getElementById('coco-status');
                const openBtn = document.getElementById('open-coco-btn');
                
                if (data.running) {
                    statusElement.innerHTML = '✅ COCO Annotator正在运行';
                    statusElement.style.color = 'green';
                    openBtn.style.display = 'inline-block';
                } else {
                    statusElement.innerHTML = '❌ COCO Annotator未运行';
                    statusElement.style.color = 'red';
                    openBtn.style.display = 'none';
                }
                
                if (data.error) {
                    statusElement.innerHTML += `<br>错误: ${data.error}`;
                }
            })
            .catch(error => {
                document.getElementById('coco-status').innerHTML = '❌ 检查状态失败: ' + error;
            });
        }
        
        function startCocoAnnotator() {
            fetch('/api/start_coco_annotator', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showCocoMessage('error', data.error);
                } else {
                    showCocoMessage('info', data.message);
                    // 等待几秒后检查状态
                    setTimeout(() => {
                        checkCocoStatus();
                    }, 5000);
                }
            })
            .catch(error => {
                showCocoMessage('error', '启动失败: ' + error);
            });
        }
        
        function stopCocoAnnotator() {
            fetch('/api/stop_coco_annotator', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showCocoMessage('error', data.error);
                } else {
                    showCocoMessage('success', data.message);
                    checkCocoStatus();
                }
            })
            .catch(error => {
                showCocoMessage('error', '停止失败: ' + error);
            });
        }
        
        function openCocoAnnotator() {
            window.open('http://localhost:5000', '_blank');
        }
        
        function showCocoMessage(type, message) {
            const messageDiv = document.getElementById('coco-message');
            messageDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
        }
        
        // ========== 3D 火柴人渲染 ==========
        // 说明：Web版本并没有直接调用test_pose3d.py的函数，而是重新实现了类似的功能
        // 对应关系：
        // - select_best_keypoints() -> web版本中的select_best_keypoints()（后端）
        // - map_2d_to_3d_centered() -> 前端updateSkeletonFromKeypoints()中使用center和scale参数实现
        // - ROI选择逻辑 -> 通过Canvas在前端实现，类似test_pose3d.py中的cv2.selectROI()
        
        let threeRenderer, threeScene, threeCamera, threeControls;
        let skeletonLine;
        let poseEventSource = null;
        let roiSelecting = false;
        let roiStartX = 0, roiStartY = 0, roiCurrentX = 0, roiCurrentY = 0;
        let roiCanvas = null, roiCtx = null, roiImage = null;
        const COCO_EDGES = [
            [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6],
            [5, 7], [7, 9],
            [6, 8], [8, 10],
            [11, 13], [13, 15],
            [12, 14], [14, 16],
            [5, 6],
            [11, 12],
            [5, 11], [6, 12]
        ];

        function ensureThree() {
            if (threeRenderer) return;
            const container = document.getElementById('three-container');
            const w = container.clientWidth;
            const h = container.clientHeight;

            threeRenderer = new THREE.WebGLRenderer({ antialias: true });
            threeRenderer.setSize(w, h);
            threeRenderer.setPixelRatio(window.devicePixelRatio);
            container.innerHTML = '';
            container.appendChild(threeRenderer.domElement);

            threeScene = new THREE.Scene();
            threeScene.background = new THREE.Color(0x111111);

            threeCamera = new THREE.PerspectiveCamera(45, w / h, 0.01, 100);
            threeCamera.position.set(0, 0, 2.2);

            threeControls = new THREE.OrbitControls(threeCamera, threeRenderer.domElement);
            threeControls.enableDamping = true;

            const light = new THREE.DirectionalLight(0xffffff, 0.8);
            light.position.set(1, 1, 2);
            threeScene.add(light);
            threeScene.add(new THREE.AmbientLight(0xffffff, 0.3));

            const material = new THREE.LineBasicMaterial({ color: 0x44ccff, linewidth: 2 });
            const geometry = new THREE.BufferGeometry();
            const maxSegments = COCO_EDGES.length;
            const positions = new Float32Array(maxSegments * 2 * 3);
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            skeletonLine = new THREE.LineSegments(geometry, material);
            threeScene.add(skeletonLine);

            const coord = new THREE.AxesHelper(0.3);
            threeScene.add(coord);

            window.addEventListener('resize', () => {
                const w2 = container.clientWidth;
                const h2 = container.clientHeight;
                threeCamera.aspect = w2 / h2;
                threeCamera.updateProjectionMatrix();
                threeRenderer.setSize(w2, h2);
            });

            function animate() {
                requestAnimationFrame(animate);
                threeControls.update();
                threeRenderer.render(threeScene, threeCamera);
            }
            animate();
        }

        function updateSkeletonFromKeypoints(people, center, scale) {
            if (!skeletonLine) return;
            const pos = skeletonLine.geometry.attributes.position.array;
            const kp = (people && people.length > 0) ? people[0] : null;
            let idx = 0;
            
            if (kp && kp.length >= 17) {
                const s = scale || 1.0;
                const cx = center && center.length >= 2 ? center[0] : 0.5;
                const cy = center && center.length >= 2 ? center[1] : 0.5;
                
                function mapPt(p) {
                    const x = (p.x - cx) * s;
                    const y = -(p.y - cy) * s;
                    const z = 0.0;
                    return [x, y, z];
                }
                
                COCO_EDGES.forEach(edge => {
                    if (edge[0] < kp.length && edge[1] < kp.length) {
                        const a = mapPt(kp[edge[0]]);
                        const b = mapPt(kp[edge[1]]);
                        pos[idx++] = a[0]; pos[idx++] = a[1]; pos[idx++] = a[2];
                        pos[idx++] = b[0]; pos[idx++] = b[1]; pos[idx++] = b[2];
                    }
                });
            } else {
                for (let i = 0; i < pos.length; i++) pos[i] = 0;
            }
            skeletonLine.geometry.attributes.position.needsUpdate = true;
        }

        function startPoseStream() {
            ensureThree();
            const videoPath = document.getElementById('video_path').value;
            const modelPath = document.getElementById('pose_model_path').value;
            const conf = parseFloat(document.getElementById('pose_conf').value || '0.5');
            const roi = document.getElementById('pose_roi').value.trim() || null;
            
            if (!videoPath) {
                setPoseStatus('error', '请填写视频路径');
                return;
            }
            
            const speed = parseFloat(document.getElementById('pose_speed').value || '1.0');
            const data = {
                video_path: videoPath,
                model_path: modelPath,
                conf_threshold: conf,
                playback_speed: speed
            };
            if (roi) {
                data.roi = roi;
            }
            
            fetch('/api/start_video_pose', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            }).then(r => r.json()).then(res => {
                if (res.error) {
                    setPoseStatus('error', res.error);
                    return;
                }
                setPoseStatus('info', '视频关键点流已启动');
                
                if (poseEventSource) {
                    poseEventSource.close();
                }
                
                poseEventSource = new EventSource('/api/pose_stream');
                poseEventSource.onmessage = (ev) => {
                    try {
                        const data = JSON.parse(ev.data);
                        if (data.type === 'keypoints') {
                            const scale = parseFloat(document.getElementById('pose_scale').value || '1.0');
                            updateSkeletonFromKeypoints(data.people, data.center, scale);
                            
                            // 显示视频帧（使用Image对象预加载以提升流畅度）
                            if (data.frame) {
                                const imgEl = document.getElementById('video_frame');
                                const placeholder = document.getElementById('video_placeholder');
                                
                                // 使用Image对象预加载，避免阻塞
                                const img = new Image();
                                img.onload = function() {
                                    imgEl.src = this.src;
                                };
                                img.src = 'data:image/jpeg;base64,' + data.frame;
                                
                                imgEl.style.display = 'block';
                                if (placeholder) placeholder.style.display = 'none';
                            }
                        } else if (data.type === 'error') {
                            setPoseStatus('error', data.message);
                        }
                    } catch (e) {
                        console.error('解析关键点数据失败:', e);
                    }
                };
                
                poseEventSource.addEventListener('end', () => {
                    setPoseStatus('success', '视频已结束');
                    if (poseEventSource) {
                        poseEventSource.close();
                        poseEventSource = null;
                    }
                });
                
                poseEventSource.onerror = () => {
                    setPoseStatus('error', 'SSE连接错误');
                };
            }).catch(err => setPoseStatus('error', '启动失败: ' + err));
        }

        function stopPoseStream() {
            if (poseEventSource) {
                poseEventSource.close();
                poseEventSource = null;
            }
            fetch('/api/stop_video_pose', { method: 'POST', headers: { 'Content-Type': 'application/json' }})
                .then(r => r.json())
                .then(res => setPoseStatus('info', res.message || '已停止'))
                .catch(() => setPoseStatus('error', '停止失败'));
        }

        function setPoseStatus(type, msg) {
            const el = document.getElementById('pose_status');
            el.innerHTML = `<div class="status ${type}">${msg}</div>`;
        }

        // ROI选择功能
        function loadVideoFrame() {
            const videoPath = document.getElementById('video_path').value;
            if (!videoPath) {
                setPoseStatus('error', '请先填写视频路径');
                return;
            }
            
            fetch('/api/get_video_first_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ video_path: videoPath })
            }).then(r => r.json()).then(res => {
                if (res.error) {
                    setPoseStatus('error', res.error);
                    return;
                }
                
                roiCanvas = document.getElementById('roi_canvas');
                roiCtx = roiCanvas.getContext('2d');
                
                const img = new Image();
                img.onload = function() {
                    const maxWidth = 800;
                    const scale = Math.min(1, maxWidth / img.width);
                    roiCanvas.width = img.width * scale;
                    roiCanvas.height = img.height * scale;
                    
                    roiCtx.drawImage(img, 0, 0, roiCanvas.width, roiCanvas.height);
                    roiImage = { img: img, scale: scale, origWidth: img.width, origHeight: img.height };
                    
                    document.getElementById('roi_canvas_container').style.display = 'block';
                    setPoseStatus('info', '请在第一帧上拖拽选择ROI区域');
                };
                img.src = res.frame;
            }).catch(err => setPoseStatus('error', '加载失败: ' + err));
        }

        // Canvas ROI选择事件
        document.addEventListener('DOMContentLoaded', function() {
            const canvas = document.getElementById('roi_canvas');
            if (!canvas) return;
            
            canvas.addEventListener('mousedown', function(e) {
                if (!roiImage) return;
                roiSelecting = true;
                const rect = canvas.getBoundingClientRect();
                roiStartX = e.clientX - rect.left;
                roiStartY = e.clientY - rect.top;
            });
            
            canvas.addEventListener('mousemove', function(e) {
                if (!roiSelecting || !roiImage) return;
                const rect = canvas.getBoundingClientRect();
                roiCurrentX = e.clientX - rect.left;
                roiCurrentY = e.clientY - rect.top;
                
                roiCtx.clearRect(0, 0, canvas.width, canvas.height);
                roiCtx.drawImage(roiImage.img, 0, 0, canvas.width, canvas.height);
                
                const w = roiCurrentX - roiStartX;
                const h = roiCurrentY - roiStartY;
                roiCtx.strokeStyle = 'red';
                roiCtx.lineWidth = 2;
                roiCtx.strokeRect(roiStartX, roiStartY, w, h);
            });
            
            canvas.addEventListener('mouseup', function(e) {
                if (!roiSelecting || !roiImage) return;
                roiSelecting = false;
            });
        });

        function confirmROI() {
            if (!roiImage) return;
            
            const x = Math.min(roiStartX, roiCurrentX);
            const y = Math.min(roiStartY, roiCurrentY);
            const w = Math.abs(roiCurrentX - roiStartX);
            const h = Math.abs(roiCurrentY - roiStartY);
            
            if (w < 10 || h < 10) {
                setPoseStatus('error', 'ROI区域太小，请重新选择');
                return;
            }
            
            // 转换为原始图片坐标
            const origX = Math.floor(x / roiImage.scale);
            const origY = Math.floor(y / roiImage.scale);
            const origW = Math.floor(w / roiImage.scale);
            const origH = Math.floor(h / roiImage.scale);
            
            document.getElementById('pose_roi').value = `${origX},${origY},${origW},${origH}`;
            document.getElementById('roi_canvas_container').style.display = 'none';
            setPoseStatus('success', `ROI已设置: ${origX},${origY},${origW},${origH}`);
        }

        function cancelROISelection() {
            document.getElementById('roi_canvas_container').style.display = 'none';
            roiImage = null;
        }
        
        // ========== 视频动作标注功能 ==========
        let videoActionData = null;
        let videoActionCurrentFrame = 0;
        let videoActionPlaying = false;
        let videoActionPlayInterval = null;
        let videoActionAnnotations = {};
        let videoActionCanvas = null;
        let videoActionCtx = null;
        let videoActionVideo = null;
        
        function startVideoActionLabeling() {
            const videoPath = document.getElementById('video_action_path').value;
            const modelPath = document.getElementById('video_action_model').value;
            const outputPath = document.getElementById('video_action_output').value || null;
            const conf = parseFloat(document.getElementById('video_action_conf').value || '0.5');
            const seqLen = parseInt(document.getElementById('video_action_seq_len').value || '30');
            const roi = document.getElementById('video_action_roi').value.trim() || null;
            
            if (!videoPath) {
                setVideoActionStatus('error', '请填写视频路径');
                return;
            }
            
            setVideoActionStatus('info', '正在提取关键点，请稍候...');
            
            fetch('/api/start_video_action_labeling', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video_path: videoPath,
                    model_path: modelPath,
                    output_path: outputPath,
                    conf_threshold: conf,
                    seq_len: seqLen,
                    roi: roi
                })
            }).then(r => r.json()).then(res => {
                if (res.error) {
                    setVideoActionStatus('error', res.error);
                    return;
                }
                setVideoActionStatus('success', '关键点提取完成！正在加载视频...');
                
                const checkInterval = setInterval(() => {
                    fetch('/api/get_video_labeling_status')
                        .then(r => r.json())
                        .then(status => {
                            if (!status.running && status.output_path) {
                                clearInterval(checkInterval);
                                loadVideoActionData(status.output_path, videoPath);
                            }
                        });
                }, 1000);
            }).catch(err => setVideoActionStatus('error', '启动失败: ' + err));
        }
        
        function loadVideoActionData(jsonPath, videoPath) {
            fetch('/api/get_file?path=' + encodeURIComponent(jsonPath))
                .then(r => r.json())
                .then(data => {
                    videoActionData = data;
                    videoActionAnnotations = {};
                    if (data.annotations) {
                        for (let start in data.annotations) {
                            const startFrame = parseInt(start);
                            const ends = data.annotations[start];
                            for (let end in ends) {
                                const endFrame = parseInt(end);
                                if (!videoActionAnnotations[startFrame]) {
                                    videoActionAnnotations[startFrame] = {};
                                }
                                videoActionAnnotations[startFrame][endFrame] = ends[end];
                            }
                        }
                    }
                    
                    videoActionVideo = document.createElement('video');
                    videoActionVideo.src = videoPath.startsWith('./') || videoPath.startsWith('/') ? 
                        '/api/get_file?path=' + encodeURIComponent(videoPath) : videoPath;
                    videoActionVideo.crossOrigin = 'anonymous';
                    videoActionVideo.preload = 'metadata';
                    
                    videoActionVideo.addEventListener('loadedmetadata', () => {
                        document.getElementById('video_action_total_frames').textContent = data.total_frames;
                        videoActionCanvas = document.getElementById('video_action_canvas');
                        videoActionCtx = videoActionCanvas.getContext('2d');
                        videoActionCanvas.width = videoActionVideo.videoWidth;
                        videoActionCanvas.height = videoActionVideo.videoHeight;
                        
                        document.getElementById('video_action_player').style.display = 'block';
                        document.getElementById('save_video_annotation_btn').style.display = 'inline-block';
                        videoActionCurrentFrame = 0;
                        updateVideoActionFrame();
                    });
                    
                    videoActionVideo.addEventListener('error', () => {
                        setVideoActionStatus('error', '无法加载视频文件。请确保视频路径正确。');
                    });
                })
                .catch(err => setVideoActionStatus('error', '加载数据失败: ' + err));
        }
        
        function updateVideoActionFrame() {
            if (!videoActionVideo || !videoActionCanvas) return;
            
            videoActionVideo.currentTime = videoActionCurrentFrame / (videoActionData.fps || 30);
            videoActionVideo.addEventListener('seeked', () => {
                videoActionCtx.drawImage(videoActionVideo, 0, 0);
                
                if (videoActionCurrentFrame < videoActionData.keypoint_sequence.length) {
                    const kp = videoActionData.keypoint_sequence[videoActionCurrentFrame];
                    if (kp && kp.length >= 34) {
                        const h = videoActionCanvas.height;
                        const w = videoActionCanvas.width;
                        
                        videoActionCtx.fillStyle = '#ffff00';
                        for (let i = 0; i < 17; i++) {
                            const x = kp[i * 2] * w;
                            const y = kp[i * 2 + 1] * h;
                            videoActionCtx.beginPath();
                            videoActionCtx.arc(x, y, 3, 0, Math.PI * 2);
                            videoActionCtx.fill();
                        }
                        
                        const edges = [[0,1],[0,2],[1,3],[2,4],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],
                                      [11,13],[13,15],[12,14],[14,16],[5,6],[11,12],[5,11],[6,12]];
                        videoActionCtx.strokeStyle = '#00c8ff';
                        videoActionCtx.lineWidth = 2;
                        edges.forEach(e => {
                            const [a, b] = e;
                            if (a < 17 && b < 17) {
                                videoActionCtx.beginPath();
                                videoActionCtx.moveTo(kp[a*2]*w, kp[a*2+1]*h);
                                videoActionCtx.lineTo(kp[b*2]*w, kp[b*2+1]*h);
                                videoActionCtx.stroke();
                            }
                        });
                    }
                }
                
                const currentAction = getVideoActionCurrentAction();
                document.getElementById('video_action_current_action').textContent = currentAction;
                document.getElementById('video_action_frame').textContent = videoActionCurrentFrame;
            }, { once: true });
        }
        
        function getVideoActionCurrentAction() {
            for (let start in videoActionAnnotations) {
                const startFrame = parseInt(start);
                if (videoActionCurrentFrame >= startFrame) {
                    const ends = videoActionAnnotations[start];
                    for (let end in ends) {
                        const endFrame = parseInt(end);
                        if (videoActionCurrentFrame <= endFrame) {
                            const actionId = ends[end];
                            const actionMap = {0: 'W-前进', 1: 'A-左', 2: 'S-后退', 3: 'D-右', 4: '空格-跳跃', 5: 'I-静止'};
                            return actionMap[actionId] || '未知';
                        }
                    }
                }
            }
            return '未标注';
        }
        
        function videoActionPlayPause() {
            videoActionPlaying = !videoActionPlaying;
            if (videoActionPlaying) {
                if (videoActionPlayInterval) clearInterval(videoActionPlayInterval);
                videoActionPlayInterval = setInterval(() => {
                    if (!videoActionPlaying) {
                        clearInterval(videoActionPlayInterval);
                        return;
                    }
                    videoActionCurrentFrame = (videoActionCurrentFrame + 1) % videoActionData.total_frames;
                    updateVideoActionFrame();
                }, 1000 / (videoActionData.fps || 30));
            } else {
                if (videoActionPlayInterval) {
                    clearInterval(videoActionPlayInterval);
                    videoActionPlayInterval = null;
                }
            }
        }
        
        function videoActionPrevFrame() {
            videoActionCurrentFrame = Math.max(0, videoActionCurrentFrame - 10);
            updateVideoActionFrame();
        }
        
        function videoActionNextFrame() {
            videoActionCurrentFrame = Math.min(videoActionData.total_frames - 1, videoActionCurrentFrame + 10);
            updateVideoActionFrame();
        }
        
        function videoActionLabel(action) {
            if (!videoActionData) return;
            
            const actionMap = {'w': 0, 'a': 1, 's': 2, 'd': 3, ' ': 4, 'idle': 5};
            const actionId = actionMap[action];
            const seqLen = parseInt(document.getElementById('video_action_seq_len').value || '30');
            const endFrame = Math.min(videoActionCurrentFrame + seqLen - 1, videoActionData.total_frames - 1);
            
            if (!videoActionAnnotations[videoActionCurrentFrame]) {
                videoActionAnnotations[videoActionCurrentFrame] = {};
            }
            videoActionAnnotations[videoActionCurrentFrame][endFrame] = actionId;
            
            setVideoActionStatus('success', `已标注: 帧 ${videoActionCurrentFrame}-${endFrame} 为 ${action.toUpperCase()}`);
            updateVideoActionFrame();
        }
        
        function saveVideoAnnotation() {
            if (!videoActionData) {
                setVideoActionStatus('error', '没有可保存的数据');
                return;
            }
            
            let outputPath = document.getElementById('video_action_output').value;
            if (!outputPath && videoActionData.video_path) {
                outputPath = videoActionData.video_path.replace(/\.[^/.]+$/, '_action_labels.json');
            }
            
            if (!outputPath) {
                setVideoActionStatus('error', '请指定输出路径');
                return;
            }
            
            const annotations = {};
            for (let start in videoActionAnnotations) {
                const startFrame = parseInt(start);
                const ends = videoActionAnnotations[start];
                annotations[startFrame] = {};
                for (let end in ends) {
                    annotations[startFrame][parseInt(end)] = ends[end];
                }
            }
            
            fetch('/api/save_video_annotation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    output_path: outputPath,
                    annotations: annotations
                })
            }).then(r => r.json()).then(res => {
                if (res.error) {
                    setVideoActionStatus('error', res.error);
                } else {
                    setVideoActionStatus('success', '标注已保存: ' + outputPath);
                }
            }).catch(err => setVideoActionStatus('error', '保存失败: ' + err));
        }
        
        function setVideoActionStatus(type, msg) {
            const el = document.getElementById('video_action_status');
            el.className = type;
            el.textContent = msg;
        }
        
        // ========== 实时游戏标注功能 ==========
        let realtimeLabelingStatusInterval = null;
        
        function startRealtimeLabeling() {
            const modelPath = document.getElementById('realtime_action_model').value;
            const monitor = document.getElementById('realtime_action_monitor').value.trim() || null;
            const conf = parseFloat(document.getElementById('realtime_action_conf').value || '0.5');
            const fps = parseFloat(document.getElementById('realtime_action_fps').value || '30');
            const outputPath = document.getElementById('realtime_action_output').value.trim() || null;
            const seqLen = parseInt(document.getElementById('realtime_action_seq_len').value || '30');
            
            setRealtimeActionStatus('info', '正在启动实时标注...');
            
            fetch('/api/start_realtime_labeling', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_path: modelPath,
                    monitor: monitor,
                    conf_threshold: conf,
                    fps: fps,
                    output_path: outputPath,
                    seq_len: seqLen
                })
            }).then(r => r.json()).then(res => {
                if (res.error) {
                    setRealtimeActionStatus('error', res.error);
                    return;
                }
                setRealtimeActionStatus('success', '实时标注已启动！请在3秒内切换到游戏窗口');
                document.getElementById('stop_realtime_btn').style.display = 'inline-block';
                document.getElementById('realtime_action_preview').style.display = 'block';
                
                realtimeLabelingStatusInterval = setInterval(() => {
                    fetch('/api/get_realtime_labeling_status')
                        .then(r => r.json())
                        .then(status => {
                            if (status.running) {
                                document.getElementById('realtime_action_state').textContent = '运行中';
                            } else {
                                clearInterval(realtimeLabelingStatusInterval);
                                document.getElementById('realtime_action_state').textContent = '已停止';
                                if (status.output_path) {
                                    setRealtimeActionStatus('success', '标注已保存: ' + status.output_path);
                                }
                            }
                        });
                }, 1000);
            }).catch(err => setRealtimeActionStatus('error', '启动失败: ' + err));
        }
        
        function stopRealtimeLabeling() {
            fetch('/api/stop_realtime_labeling', { method: 'POST' })
                .then(r => r.json())
                .then(res => {
                    setRealtimeActionStatus('info', res.message || '停止指令已发送');
                    if (realtimeLabelingStatusInterval) {
                        clearInterval(realtimeLabelingStatusInterval);
                        realtimeLabelingStatusInterval = null;
                    }
                })
                .catch(err => setRealtimeActionStatus('error', '停止失败: ' + err));
        }
        
        function setRealtimeActionStatus(type, msg) {
            const el = document.getElementById('realtime_action_status');
            el.className = type;
            el.textContent = msg;
        }
        
        // 页面加载时自动加载数据集信息
        window.onload = function() {
            loadDatasetInfo();
            loadLogs();
            checkCocoStatus();
        };
    </script>
</body>
</html>
    '''
        
        with open('templates/index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        logging.info('index.html已生成')
    else:
        logging.info('index.html已存在，跳过生成（保留用户修改）')
    
    
    app.run(host='0.0.0.0', port=8080, debug=False)
