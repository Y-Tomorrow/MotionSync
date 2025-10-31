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
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
import threading

# 导入现有功能模块
from auto_label import auto_label_yolo_format
from txt_coco_json import txt_to_coco, coco_to_txt
from txt_json import yolo_pose_txt_to_labelme, labelme_to_yolo_pose
from train_pose import train_yolov8_pose

app = Flask(__name__)

# 全局变量存储任务状态
task_status = {
    'running': False,
    'progress': 0,
    'message': '',
    'logs': []
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
        </div>
    </div>

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
