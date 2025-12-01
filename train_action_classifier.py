#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练动作分类模型
从标注的JSON文件中加载数据，训练动作分类器
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
from pathlib import Path
import matplotlib
# 设置非GUI后端，避免在后台线程中使用GUI导致崩溃
# 必须在导入pyplot之前设置，并且要强制设置
import os
os.environ['MPLBACKEND'] = 'Agg'  # 通过环境变量强制设置
matplotlib.use('Agg', force=True)  # force=True 强制设置，即使已经导入过
import matplotlib.pyplot as plt
# 确保使用非交互式后端
plt.ioff()  # 关闭交互模式

# seaborn为可选依赖
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("警告: seaborn未安装，混淆矩阵将使用matplotlib绘制。建议安装: pip install seaborn")

try:
    from action_classifier import ActionClassifier, ActionClassifierGRU, ActionClassifierTransformer, ActionClassifierSTGCN
    print("✓ 成功导入动作分类模型", flush=True)
except ImportError as e:
    print(f"✗ 导入动作分类模型失败: {e}", flush=True)
    raise


class ActionDataset(Dataset):
    """动作数据集"""
    
    def __init__(self, sequences, labels, sequence_length=30):
        """
        Args:
            sequences: 关键点序列列表，每个元素是 (seq_len, 34) 的数组
            labels: 标签列表，每个元素是动作类别ID
            sequence_length: 序列长度
        """
        import sys
        self.sequences = []
        self.labels = []
        
        total = len(sequences)
        print(f"    总共需要处理 {total} 个序列", flush=True)
        sys.stdout.flush()
        
        for idx, (seq, label) in enumerate(zip(sequences, labels)):
            # 确保seq是numpy数组
            if not isinstance(seq, np.ndarray):
                seq = np.array(seq, dtype=np.float32)
            
            # 确保序列长度一致
            seq_len = seq.shape[0] if len(seq.shape) > 0 else len(seq)
            
            if seq_len < sequence_length:
                # 如果序列太短，重复最后一帧
                if isinstance(seq, np.ndarray):
                    last_frame = seq[-1:]
                    padding = np.repeat(last_frame, sequence_length - seq_len, axis=0)
                    seq = np.concatenate([seq, padding], axis=0)
                else:
                    padding = [seq[-1]] * (sequence_length - seq_len)
                    seq = seq + padding
            elif seq_len > sequence_length:
                # 如果序列太长，截取
                seq = seq[:sequence_length]
            
            self.sequences.append(seq)
            self.labels.append(label)
            
            # 每处理10%的数据输出一次进度
            if (idx + 1) % max(1, total // 10) == 0 or (idx + 1) == total:
                print(f"    进度: {idx + 1}/{total} ({100 * (idx + 1) / total:.1f}%)", flush=True)
                sys.stdout.flush()
        
        print(f"    正在转换为numpy数组...", flush=True)
        sys.stdout.flush()
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        print(f"    转换完成，最终形状: {self.sequences.shape}", flush=True)
        sys.stdout.flush()
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]])[0]


def load_annotations(json_path):
    """从JSON文件加载标注数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    keypoint_sequence = data['keypoint_sequence']
    annotations = data['annotations']
    action_map = data.get('action_map', {})
    
    # 将标注转换为序列和标签
    sequences = []
    labels = []
    
    # 按开始帧排序
    sorted_starts = sorted([int(k) for k in annotations.keys()])
    
    print(f"  找到 {len(sorted_starts)} 个标注片段")
    
    for start_frame in sorted_starts:
        start_dict = annotations[str(start_frame)]
        for end_frame_str, action_id in start_dict.items():
            end_frame = int(end_frame_str)
            
            # 提取该片段的序列
            if start_frame < len(keypoint_sequence) and end_frame < len(keypoint_sequence):
                seq = keypoint_sequence[start_frame:end_frame+1]
                if len(seq) > 0:
                    # 将关键点列表转换为numpy数组
                    seq_array = np.array(seq, dtype=np.float32)
                    
                    sequences.append(seq_array)
                    labels.append(int(action_id))
    
    print(f"  提取了 {len(sequences)} 个训练样本")
    return sequences, labels, action_map


def load_multiple_annotations(json_dir):
    """从目录中加载多个标注文件"""
    all_sequences = []
    all_labels = []
    action_map = None
    
    json_files = list(Path(json_dir).glob('*_action_labels.json'))
    print(f"找到 {len(json_files)} 个标注文件")
    
    for json_path in json_files:
        print(f"加载: {json_path}")
        sequences, labels, am = load_annotations(str(json_path))
        all_sequences.extend(sequences)
        all_labels.extend(labels)
        if action_map is None:
            action_map = am
    
    return all_sequences, all_labels, action_map


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    import sys
    batch_count = 0
    total_batches = len(dataloader)
    
    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        batch_count += 1
        # 每5个batch或最后一个batch输出一次进度
        if batch_count % 5 == 0 or batch_count == total_batches:
            print(f"  训练进度: {batch_count}/{total_batches} batches", flush=True)
            sys.stdout.flush()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy, all_preds, all_labels


def train_action_classifier(
    data_dir,
    model_type='lstm',
    sequence_length=30,
    hidden_dim=128,
    num_layers=2,
    batch_size=32,
    epochs=50,
    learning_rate=0.001,
    device='cuda',
    output_dir='./models/action_classifier'
):
    """训练动作分类器"""
    
    # 加载数据
    print("加载数据...")
    if os.path.isdir(data_dir):
        sequences, labels, action_map = load_multiple_annotations(data_dir)
    else:
        sequences, labels, action_map = load_annotations(data_dir)
    
    print(f"加载了 {len(sequences)} 个样本")
    print(f"动作类别: {action_map}")
    
    # 统计类别分布
    if len(labels) == 0:
        print("错误: 没有加载到任何标注数据！")
        return
    
    # 将labels转换为numpy数组进行统计
    labels_array = np.array(labels)
    unique, counts = np.unique(labels_array, return_counts=True)
    print("\n类别分布:")
    action_names = {v: k for k, v in action_map.items()}
    for label_id, count in zip(unique, counts):
        action_name = action_names.get(int(label_id), f"class_{int(label_id)}")
        print(f"  {action_name}: {count} ({count/len(labels)*100:.1f}%)")
    
    # 检查数据平衡性
    if len(unique) < 2:
        print("\n警告: 数据不平衡！只有一个动作类别，训练效果可能不佳。")
        print("建议: 收集更多不同动作的标注数据。")
    elif len(unique) > 1:
        max_count = counts.max()
        min_count = counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        if imbalance_ratio > 5:
            print(f"\n警告: 数据不平衡！最大类别与最小类别比例为 {imbalance_ratio:.1f}:1")
            print("建议: 使用类别权重或收集更多少数类别的数据。")
    
    # 划分训练集和验证集
    # 如果只有一个类别，不能使用stratify
    if len(unique) < 2:
        print("\n警告: 只有一个动作类别，无法进行有效训练！")
        print("请确保标注数据包含多种动作（W/A/S/D/空格/静止）。")
        print("训练已取消，请重新标注包含多种动作的数据。")
        return
    
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, labels, test_size=0.2, random_state=42, stratify=labels
        )
    except ValueError as e:
        # 如果stratify失败（某些类别样本太少），不使用stratify
        print(f"警告: 无法使用分层划分 ({e})，使用随机划分")
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )
    
    print(f"\n训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    
    # 创建数据集和数据加载器
    print("\n正在创建数据集...", flush=True)
    import sys
    sys.stdout.flush()
    
    print(f"  处理训练集 ({len(X_train)} 样本)...", flush=True)
    sys.stdout.flush()
    train_dataset = ActionDataset(X_train, y_train, sequence_length)
    print(f"  训练集创建完成", flush=True)
    sys.stdout.flush()
    
    print(f"  处理验证集 ({len(X_val)} 样本)...", flush=True)
    sys.stdout.flush()
    val_dataset = ActionDataset(X_val, y_val, sequence_length)
    print("数据集创建完成", flush=True)
    sys.stdout.flush()
    
    print("正在创建数据加载器...", flush=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("数据加载器创建完成", flush=True)
    
    # 创建模型
    print(f"\n正在创建模型 ({model_type})...", flush=True)
    num_classes = len(action_map)
    input_dim = 34  # 17个关键点 * 2坐标
    
    if model_type == 'lstm':
        model = ActionClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=0.3
        )
    elif model_type == 'gru':
        model = ActionClassifierGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=0.3
        )
    elif model_type == 'transformer':
        model = ActionClassifierTransformer(
            input_dim=input_dim,
            d_model=hidden_dim,
            num_classes=num_classes,
            dropout=0.3
        )
    elif model_type == 'stgcn':
        # ST-GCN使用不同的参数
        print(f"    创建ST-GCN模型: num_nodes=17, in_channels=2, num_classes={num_classes}", flush=True)
        model = ActionClassifierSTGCN(
            num_nodes=17,  # COCO pose 17个关键点
            in_channels=2,  # x, y坐标
            num_classes=num_classes,
            dropout=0.3
        )
        print(f"    ST-GCN模型创建完成", flush=True)
    else:
        raise ValueError(f"未知的模型类型: {model_type}，支持的类型: lstm, gru, transformer, stgcn")
    
    print(f"正在将模型移动到设备: {device}...", flush=True)
    model = model.to(device)
    print(f"模型已移动到设备: {device}", flush=True)
    
    print(f"\n模型: {model_type}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"设备: {device}")
    
    # 损失函数和优化器
    print("正在初始化损失函数和优化器...", flush=True)
    # 如果数据不平衡，使用类别权重
    if len(unique) > 1:
        # 为所有类别计算权重（包括没有样本的类别）
        # num_classes是模型的总类别数，可能大于实际数据中的类别数
        class_counts = np.bincount(labels_array, minlength=num_classes)
        total = len(labels_array)
        
        # 避免除零：如果某个类别没有样本，使用平滑处理
        class_counts = class_counts.astype(np.float32)
        class_counts[class_counts == 0] = 0.5  # 平滑处理，避免权重过大
        
        class_weights = total / (len(unique) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(device)
        
        # 显示权重信息
        weight_dict = {}
        for i in range(num_classes):
            action_name = action_names.get(i, f'class_{i}')
            count = int(class_counts[i])
            weight_dict[action_name] = {
                'weight': float(class_weights[i].cpu().numpy()),
                'count': count
            }
        print(f"使用类别权重:")
        for name, info in weight_dict.items():
            print(f"  {name}: weight={info['weight']:.4f}, count={info['count']}")
        sys.stdout.flush()
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    print("优化器初始化完成", flush=True)
    
    # 训练
    print(f"正在创建输出目录: {output_dir}...", flush=True)
    os.makedirs(output_dir, exist_ok=True)
    print("输出目录准备完成", flush=True)
    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("\n开始训练...")
    print("=" * 60)
    import sys
    sys.stdout.flush()  # 确保输出立即显示
    
    # 更新全局状态（如果通过web调用）
    try:
        import web_app
        if hasattr(web_app, 'action_training_state'):
            web_app.action_training_state['current_epoch'] = 0
    except:
        pass
    
    for epoch in range(epochs):
        # 更新epoch状态
        try:
            import web_app
            if hasattr(web_app, 'action_training_state'):
                web_app.action_training_state['current_epoch'] = epoch + 1
        except:
            pass
        
        print(f"\n[Epoch {epoch+1}/{epochs}] 开始训练阶段...", flush=True)
        sys.stdout.flush()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        print(f"[Epoch {epoch+1}/{epochs}] 开始验证阶段...", flush=True)
        sys.stdout.flush()
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{epochs} 完成")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        sys.stdout.flush()  # 确保输出立即显示
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(output_dir, f'best_{model_type}_action_classifier.pth')
            save_dict = {
                'model_state_dict': model.state_dict(),
                'model_type': model_type,
                'num_classes': num_classes,
                'sequence_length': sequence_length,
                'action_map': action_map,
                'epoch': epoch,
                'val_acc': val_acc
            }
            
            # 根据模型类型保存不同的参数
            if model_type == 'stgcn':
                save_dict.update({
                    'num_nodes': 17,
                    'in_channels': 2
                })
            else:
                save_dict.update({
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers
                })
            
            torch.save(save_dict, model_path)
            print(f"  ✓ 保存最佳模型 (Val Acc: {val_acc:.2f}%)")
            sys.stdout.flush()
    
    # 绘制训练曲线
    # 确保关闭所有已打开的figure，避免GUI相关错误
    plt.close('all')
    fig = plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    curves_path = os.path.join(output_dir, f'training_curves_{model_type}.png')
    plt.savefig(curves_path, dpi=100, bbox_inches='tight')
    plt.close(fig)  # 立即关闭figure，释放资源
    print(f"\n训练曲线已保存: {curves_path}")
    
    # 最终评估
    print("\n最终评估:")
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    # 分类报告
    # 只使用实际存在的类别
    unique_labels = sorted(set(list(val_labels) + list(val_preds)))
    action_names_list = [action_names.get(i, f"class_{i}") for i in unique_labels]
    
    print("\n分类报告:")
    try:
        # 使用labels参数指定实际存在的类别
        print(classification_report(val_labels, val_preds, 
                                  labels=unique_labels,
                                  target_names=action_names_list,
                                  zero_division=0))
    except Exception as e:
        print(f"生成分类报告时出错: {e}")
        # 如果还是出错，使用简化版本
        print(f"实际类别: {unique_labels}")
        print(f"预测类别: {sorted(set(val_preds))}")
        from sklearn.metrics import accuracy_score
        print(f"准确率: {accuracy_score(val_labels, val_preds):.2%}")
    
    # 混淆矩阵
    # 只使用实际存在的类别
    unique_labels = sorted(set(list(val_labels) + list(val_preds)))
    cm = confusion_matrix(val_labels, val_preds, labels=unique_labels)
    action_names_list = [action_names.get(i, f"class_{i}") for i in unique_labels]
    
    plt.close('all')  # 确保关闭之前的figure
    fig = plt.figure(figsize=(8, 6))
    
    if HAS_SEABORN:
        # 使用seaborn绘制（更美观）
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=action_names_list, yticklabels=action_names_list)
    else:
        # 使用matplotlib绘制（无需seaborn）
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        tick_marks = np.arange(len(action_names_list))
        plt.xticks(tick_marks, action_names_list, rotation=45)
        plt.yticks(tick_marks, action_names_list)
        
        # 添加数值标注
        thresh = cm.max() / 2. if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f'confusion_matrix_{model_type}.png')
    plt.savefig(cm_path, dpi=100, bbox_inches='tight')
    plt.close(fig)  # 立即关闭figure，释放资源
    print(f"混淆矩阵已保存: {cm_path}")
    
    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    print(f"模型保存在: {os.path.join(output_dir, f'best_{model_type}_action_classifier.pth')}")


def main():
    parser = argparse.ArgumentParser(description='训练动作分类模型')
    parser.add_argument('--data', type=str, required=True, 
                       help='标注JSON文件或包含多个JSON文件的目录')
    parser.add_argument('--model_type', type=str, default='lstm', 
                       choices=['lstm', 'gru', 'transformer', 'stgcn'],
                       help='模型类型 (lstm/gru/transformer/stgcn)')
    parser.add_argument('--seq_len', type=int, default=30, help='序列长度')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM/GRU层数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--output', type=str, default='./models/action_classifier', 
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 检查设备
    print(f"检查设备: {args.device}...", flush=True)
    if args.device == 'cuda':
        if torch.cuda.is_available():
            print(f"CUDA可用，使用GPU: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            print("CUDA不可用，切换到CPU", flush=True)
            args.device = 'cpu'
    else:
        print(f"使用CPU", flush=True)
    
    train_action_classifier(
        data_dir=args.data,
        model_type=args.model_type,
        sequence_length=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()


