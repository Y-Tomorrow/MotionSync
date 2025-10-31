# MotionSync - 位姿提取与动作估计软件

## 🎯 功能特性

- **位姿提取**: 使用YOLOv8-pose模型提取人体关键点位姿信息
- **动作估计**: 基于关键点序列进行动作识别和估计
- **格式转换**: 支持TXT ↔ COCO JSON ↔ LabelMe JSON转换
- **模型训练**: 基于现有数据训练自定义位姿检测模型
- **Web界面**: 现代化浏览器界面，支持实时进度显示

## 🚀 快速开始

### 1. 环境准备
```bash
# 创建conda环境
conda create -n yolo python=3.10
conda activate yolo

# 安装依赖
pip install torch torchvision torchaudio
pip install ultralytics flask PyQt5
```

### 2. 启动Web版本
```bash
conda activate yolo
python web_app.py
```

### 3. 访问界面
打开浏览器访问: http://localhost:8080

## 🌐 Web界面使用

### 位姿提取
1. 点击"位姿提取"标签页
2. 设置参数：
   - 图片目录: `./datasets/images/train`
   - 模型路径: `./models/yolov8n-pose.pt`
   - 输出目录: `./datasets/labels/train`
   - 置信度阈值: `0.5`
3. 点击"开始位姿提取"
4. 观察进度条和日志

### 格式转换
1. 点击"格式转换"标签页
2. 选择转换类型
3. 设置输入和输出路径
4. 点击"开始转换"

### JSON验证与可视化
1. 点击"JSON验证"标签页
2. 输入JSON文件路径
3. 点击"验证JSON"检查文件结构
4. 点击"可视化关键点"查看关键点信息

### COCO Annotator管理
1. 点击"COCO Annotator"标签页
2. 点击"启动COCO Annotator"启动服务
3. 等待启动完成后，点击"打开COCO Annotator"
4. 在新窗口中验证和修改JSON文件
5. 完成后点击"停止COCO Annotator"释放资源

### 数据集管理
1. 点击"数据集信息"标签页
2. 查看数据集统计信息
3. 点击"刷新信息"更新数据

## 📁 项目结构

```
sofatware/
├── web_app.py              # Web版本主程序 ⭐
├── start_web.sh            # Web版本启动脚本
├── auto_label.py            # 自动标定模块
├── txt_coco_json.py         # TXT与COCO JSON转换
├── txt_json.py              # TXT与LabelMe JSON转换
├── train_pose.py            # 模型训练
├── requirements.txt         # 依赖包列表
├── README.md                # 使用说明
├── WEB_GUIDE.md             # Web版本详细指南
├── datasets/                # 数据集目录
│   ├── images/
│   │   ├── train/           # 训练图片
│   │   └── val/             # 验证图片
│   ├── labels/
│   │   ├── train/           # 训练标签
│   │   └── val/             # 验证标签
│   └── train.yaml           # 数据集配置
├── models/                  # 模型文件
│   ├── yolov8n-pose.pt      # 预训练模型
│   ├── yolov8l-pose.pt      # 大模型
│   └── yolo11n.pt           # YOLO11模型
└── templates/               # Web界面模板
    └── index.html           # 主页面
```

## 🔧 命令行使用

如果需要在命令行中使用，可以直接调用功能模块：

```bash
# 位姿提取
python -c "from auto_label import auto_label_yolo_format; auto_label_yolo_format('./datasets/images/train', './models/yolov8n-pose.pt', './datasets/labels/train', 0.5)"

# 格式转换
python -c "from txt_coco_json import txt_to_coco; txt_to_coco('./datasets/images/train', './datasets/labels/train', './output.json')"
```

## 📊 数据集格式

### YOLO格式
```
class_id x_center y_center width height kpt1_x kpt1_y kpt1_v ...
```

### 关键点定义 (17个关键点)
```
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
```

## 🎉 总结

**推荐使用Web版本**，它提供了：
- 🌐 现代化浏览器界面
- ⚡ 实时进度显示
- 🔧 完整位姿提取和动作估计功能
- 📱 跨平台访问

**立即开始**: `python web_app.py` 然后访问 http://localhost:8080