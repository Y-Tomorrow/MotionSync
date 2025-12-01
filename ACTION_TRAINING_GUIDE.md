# 视频动作识别训练指南

本指南将帮助您训练一个能够识别视频中人物动作（W/A/S/D/空格跳跃）的模型。

## 📋 流程概览

1. **视频标注** - 从视频中提取关键点并标注动作
2. **实时游戏标注** - 捕获游戏画面，通过键盘操作实时标注（推荐）
3. **模型训练** - 训练动作分类模型
4. **动作识别** - 使用训练好的模型识别视频中的动作

## 🚀 快速开始

### 步骤1: 选择标注方式

#### 方式A: 实时游戏画面标注（推荐）

使用 `realtime_game_labeler.py` 在玩游戏时实时标注动作。这种方式更自然，标注效率更高。

```bash
# 全屏捕获
python realtime_game_labeler.py \
    --model ./models/yolov8n-pose.pt \
    --conf 0.5 \
    --fps 30

# 指定屏幕区域（例如游戏窗口位置）
python realtime_game_labeler.py \
    --model ./models/yolov8n-pose.pt \
    --monitor 0,0,1920,1080 \
    --conf 0.5 \
    --fps 30 \
    --output ./game_labels.json
```

**操作说明：**
- 运行脚本后，3秒倒计时，请切换到游戏窗口
- **按住W/A/S/D/空格** - 进行对应动作，系统自动标注
- **ESC** - 停止标注并保存
- 系统会实时显示检测到的pose和当前动作

**参数说明：**
- `--model`: YOLO pose模型路径
- `--monitor`: 屏幕区域，格式 `x,y,width,height`（可选，不指定则全屏）
- `--conf`: 置信度阈值（默认0.5）
- `--fps`: 捕获帧率（默认30）
- `--seq_len`: 序列长度（默认30，用于后续训练）
- `--output`: 输出JSON路径（可选，默认带时间戳）

**优势：**
- ✅ 标注过程自然，边玩边标注
- ✅ 数据质量高，动作与键盘输入完全同步
- ✅ 效率高，无需手动逐帧标注
- ✅ 支持实时预览pose检测结果

**注意事项：**
- 需要安装额外依赖：`pip install mss pynput`
- 在Linux上可能需要权限：`sudo pip install pynput` 或使用 `sudo` 运行
- 建议使用全屏游戏或指定游戏窗口区域以提高性能

#### 方式B: 视频动作标注

### 步骤2: 视频动作标注

使用 `video_action_labeler.py` 从视频中提取关键点并标注动作。

```bash
python video_action_labeler.py \
    --video ./1.mp4 \
    --model ./models/yolov8n-pose.pt \
    --output ./1_action_labels.json \
    --conf 0.5 \
    --seq_len 30
```

**操作说明：**
- **W** - 标注为前进
- **A** - 标注为左移
- **S** - 标注为后退
- **D** - 标注为右移
- **空格** - 标注为跳跃
- **I** - 标注为静止
- **←/→** - 前进/后退10帧
- **P** - 播放/暂停
- **Q** - 退出并保存
- **R** - 重置当前片段标注

**参数说明：**
- `--video`: 输入视频路径
- `--model`: YOLO pose模型路径
- `--output`: 输出JSON标注文件路径
- `--conf`: 置信度阈值（默认0.5）
- `--seq_len`: 标注序列长度，即每次标注的帧数（默认30）
- `--roi`: ROI区域，格式 `x,y,w,h`（可选，用于提高识别速度）

**标注技巧：**
1. 先播放视频，找到包含动作的片段
2. 暂停在动作开始帧
3. 按对应按键（W/A/S/D/空格/I）标注动作
4. 系统会自动标注从当前帧开始的30帧（可通过`--seq_len`调整）
5. 可以多次标注同一个视频的不同片段

### 步骤3: 训练动作分类模型

收集多个标注文件后，使用 `train_action_classifier.py` 训练模型。

**单个标注文件训练：**
```bash
python train_action_classifier.py \
    --data ./1_action_labels.json \
    --model_type lstm \
    --seq_len 30 \
    --hidden_dim 128 \
    --num_layers 2 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --device cuda \
    --output ./models/action_classifier
```

**多个标注文件训练（推荐）：**
```bash
# 将所有标注JSON文件放在一个目录中
python train_action_classifier.py \
    --data ./action_labels/ \
    --model_type lstm \
    --epochs 50 \
    --device cuda
```

**参数说明：**
- `--data`: 标注JSON文件或包含多个JSON文件的目录
- `--model_type`: 模型类型，可选 `lstm`、`gru`、`transformer`（默认lstm）
- `--seq_len`: 序列长度，需与标注时一致（默认30）
- `--hidden_dim`: 隐藏层维度（默认128）
- `--num_layers`: LSTM/GRU层数（默认2）
- `--batch_size`: 批次大小（默认32）
- `--epochs`: 训练轮数（默认50）
- `--lr`: 学习率（默认0.001）
- `--device`: 设备，`cuda` 或 `cpu`（默认cuda）
- `--output`: 模型输出目录（默认`./models/action_classifier`）

**模型类型选择：**
- **LSTM**: 平衡性能和准确率，推荐使用
- **GRU**: 更轻量，训练更快
- **Transformer**: 更强大但需要更多数据和计算资源
- **ST-GCN**: 基于图卷积网络，专门为动作识别设计，能更好地捕捉骨架的空间结构关系，适合动作识别任务

**训练输出：**
- 最佳模型：`best_{model_type}_action_classifier.pth`
- 训练曲线：`training_curves_{model_type}.png`
- 混淆矩阵：`confusion_matrix_{model_type}.png`

### 步骤4: 动作识别推理

使用训练好的模型识别视频中的动作。

```bash
python infer_action.py \
    --video ./1.mp4 \
    --pose_model ./models/yolov8n-pose.pt \
    --action_model ./models/action_classifier/best_lstm_action_classifier.pth \
    --conf 0.5 \
    --device cuda \
    --output ./output_with_actions.mp4
```

**参数说明：**
- `--video`: 输入视频路径
- `--pose_model`: YOLO pose模型路径
- `--action_model`: 训练好的动作分类模型路径
- `--conf`: 置信度阈值（默认0.5）
- `--roi`: ROI区域，格式 `x,y,w,h`（可选）
- `--device`: 设备，`cuda` 或 `cpu`（默认cuda）
- `--output`: 输出视频路径（可选，不指定则只显示不保存）

**操作说明：**
- 按 **Q** 退出

## 📊 数据格式

### 标注JSON格式

```json
{
  "video_path": "./1.mp4",
  "total_frames": 1000,
  "fps": 30.0,
  "keypoint_sequence": [
    [0.5, 0.3, 0.6, 0.4, ...],  // 17个关键点，每个点2个坐标，共34个值
    ...
  ],
  "frame_indices": [0, 1, 2, ...],
  "annotations": {
    "0": {
      "29": 0  // 帧0-29标注为动作0（W前进）
    },
    "100": {
      "129": 1  // 帧100-129标注为动作1（A左移）
    }
  },
  "action_map": {
    "w": 0,
    "a": 1,
    "s": 2,
    "d": 3,
    " ": 4,
    "idle": 5
  }
}
```

## 💡 最佳实践

### 1. 数据收集
- **多样性**: 收集不同场景、不同人物的动作视频
- **平衡性**: 尽量保证每个动作类别的样本数量相近
- **质量**: 确保视频中人物清晰可见，关键点检测准确

### 2. 标注技巧
- **序列长度**: 根据动作持续时间调整`--seq_len`，一般30帧（1秒@30fps）比较合适
- **重叠标注**: 可以重叠标注片段，增加训练数据
- **边界处理**: 在动作开始和结束的边界处仔细标注

### 3. 模型训练
- **数据量**: 建议每个动作类别至少50-100个样本
- **验证集**: 系统自动划分20%作为验证集
- **过拟合**: 如果验证准确率远低于训练准确率，考虑增加dropout或减少模型复杂度
- **学习率**: 如果损失不下降，尝试降低学习率

### 4. 模型选择
- **小数据集** (<500样本): 使用GRU
- **中等数据集** (500-2000样本): 使用LSTM或ST-GCN
- **大数据集** (>2000样本): 可以尝试Transformer或ST-GCN
- **动作识别任务**: 推荐使用ST-GCN，因为它专门为骨架动作识别设计

## 🔧 故障排除

### 问题1: 标注时无法检测到关键点
- **解决**: 降低`--conf`阈值，或使用`--roi`指定人物区域

### 问题2: 训练时内存不足
- **解决**: 减小`--batch_size`或`--hidden_dim`

### 问题3: 识别准确率低
- **解决**: 
  - 增加训练数据
  - 调整序列长度`--seq_len`
  - 尝试不同的模型类型
  - 检查标注质量

### 问题4: CUDA内存不足
- **解决**: 使用`--device cpu`或减小批次大小

## 📁 项目结构

```
sofatware/
├── video_action_labeler.py      # 视频动作标注工具
├── realtime_game_labeler.py     # 实时游戏画面标注工具 ⭐推荐
├── action_classifier.py          # 动作分类模型定义
├── train_action_classifier.py    # 训练脚本
├── infer_action.py               # 推理脚本
├── ACTION_TRAINING_GUIDE.md      # 本指南
├── models/
│   ├── yolov8n-pose.pt          # Pose检测模型
│   └── action_classifier/       # 训练好的动作分类模型
│       ├── best_lstm_action_classifier.pth
│       ├── training_curves_lstm.png
│       └── confusion_matrix_lstm.png
└── action_labels/               # 标注数据目录（可选）
    ├── game1_action_labels.json  # 实时游戏标注数据
    ├── video1_action_labels.json
    └── ...
```

## 🎯 示例工作流

### 工作流A: 实时游戏标注（推荐）

```bash
# 1. 实时标注游戏动作（多次运行收集不同场景）
python realtime_game_labeler.py --model ./models/yolov8n-pose.pt --output ./labels/game1.json
python realtime_game_labeler.py --model ./models/yolov8n-pose.pt --output ./labels/game2.json
python realtime_game_labeler.py --model ./models/yolov8n-pose.pt --output ./labels/game3.json

# 2. 训练模型
python train_action_classifier.py --data ./labels/ --model_type lstm --epochs 50

# 3. 测试识别
python infer_action.py \
    --video ./test_video.mp4 \
    --action_model ./models/action_classifier/best_lstm_action_classifier.pth \
    --output ./test_result.mp4
```

### 工作流B: 视频标注

```bash
# 1. 标注第一个视频
python video_action_labeler.py --video ./videos/walk1.mp4 --output ./action_labels/walk1.json

# 2. 标注更多视频
python video_action_labeler.py --video ./videos/jump1.mp4 --output ./action_labels/jump1.json
python video_action_labeler.py --video ./videos/run1.mp4 --output ./action_labels/run1.json

# 3. 训练模型（支持lstm/gru/transformer/stgcn）
python train_action_classifier.py --data ./action_labels/ --model_type stgcn --epochs 50

# 4. 测试识别
python infer_action.py \
    --video ./videos/test.mp4 \
    --action_model ./models/action_classifier/best_lstm_action_classifier.pth \
    --output ./test_result.mp4
```

## 📝 注意事项

1. **序列长度一致性**: 标注时的`--seq_len`必须与训练时的`--seq_len`一致
2. **模型兼容性**: 推理时使用的模型必须与训练时的模型类型匹配
3. **关键点格式**: 系统使用COCO 17点关键点格式
4. **视频格式**: 支持常见视频格式（mp4, avi等）
5. **实时标注权限**: 在Linux上运行实时标注工具可能需要sudo权限（用于键盘监听）
6. **屏幕捕获性能**: 使用`mss`库可以获得更好的屏幕捕获性能，建议安装：`pip install mss`
7. **ST-GCN使用现有pose数据**: ST-GCN可以使用YOLO提取的pose关键点数据训练，不需要自己提取pose。只需使用标注工具（视频标注或实时标注）提取关键点序列，然后选择`--model_type stgcn`进行训练即可。

## 🎉 完成！

现在您已经掌握了完整的视频动作识别训练流程。开始标注您的第一个视频吧！

如有问题，请检查：
- 依赖包是否安装完整（torch, ultralytics, opencv-python等）
- 模型文件路径是否正确
- 视频文件是否可以正常打开

