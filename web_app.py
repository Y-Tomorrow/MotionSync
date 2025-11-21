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
    'model': None
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
            try:
                logging.info('加载YOLO姿态模型中...')
                model = YOLO(model_path)
                pose_stream_state['model'] = model
                logging.info('模型加载完成，开始读取视频')
                
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

if __name__ == '__main__':
    setup_logging()
    
    # 创建templates目录和HTML文件
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
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
    
    
    app.run(host='0.0.0.0', port=8080, debug=False)
