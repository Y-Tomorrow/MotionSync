#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作分类模型
基于LSTM/GRU的时序动作分类器
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionClassifier(nn.Module):
    """基于LSTM的动作分类模型"""
    
    def __init__(self, input_dim=34, hidden_dim=128, num_layers=2, num_classes=6, dropout=0.3):
        """
        Args:
            input_dim: 输入维度 (17个关键点 * 2坐标 = 34)
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            num_classes: 动作类别数 (W/A/S/D/空格/静止 = 6)
            dropout: Dropout比率
        """
        super(ActionClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2因为双向LSTM
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        # lstm_out: (batch_size, seq_len, hidden_dim * 2)
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # 全连接层
        x = self.fc1(last_output)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ActionClassifierGRU(nn.Module):
    """基于GRU的动作分类模型（更轻量）"""
    
    def __init__(self, input_dim=34, hidden_dim=128, num_layers=2, num_classes=6, dropout=0.3):
        super(ActionClassifierGRU, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        gru_out, h_n = self.gru(x)
        last_output = gru_out[:, -1, :]
        
        x = self.fc1(last_output)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ActionClassifierTransformer(nn.Module):
    """基于Transformer的动作分类模型（更强大但更复杂）"""
    
    def __init__(self, input_dim=34, d_model=128, nhead=8, num_layers=3, num_classes=6, dropout=0.3):
        super(ActionClassifierTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 分类头
        self.fc1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        # 投影到d_model维度
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # 使用最后一个时间步或平均池化
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # 分类
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class PositionalEncoding(nn.Module):
    """位置编码（用于Transformer）"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class STGCNBlock(nn.Module):
    """ST-GCN基本块：空间图卷积 + 时间卷积"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, residual=True):
        super(STGCNBlock, self).__init__()
        
        self.kernel_size = kernel_size
        self.residual = residual
        
        # 空间图卷积（使用1x1卷积近似）
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 时间卷积
        self.temporal_conv = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=(kernel_size, 1), 
            stride=(stride, 1),
            padding=((kernel_size - 1) // 2, 0)
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 残差连接
        if residual:
            if in_channels != out_channels or stride != 1:
                self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1))
                self.residual_bn = nn.BatchNorm2d(out_channels)
            else:
                self.residual_conv = None
                self.residual_bn = None
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, time, nodes)
        """
        residual = x
        
        # 空间图卷积
        out = self.spatial_conv(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # 时间卷积
        out = self.temporal_conv(out)
        out = self.bn2(out)
        
        # 残差连接
        if self.residual:
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
                residual = self.residual_bn(residual)
            out = out + residual
        
        out = F.relu(out)
        return out


class ActionClassifierSTGCN(nn.Module):
    """基于ST-GCN的动作分类模型"""
    
    def __init__(self, num_nodes=17, in_channels=2, num_classes=6, dropout=0.3):
        """
        Args:
            num_nodes: 关键点数量 (COCO pose = 17)
            in_channels: 每个关键点的特征维度 (x, y = 2)
            num_classes: 动作类别数
            dropout: Dropout比率
        """
        super(ActionClassifierSTGCN, self).__init__()
        
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # 输入投影
        self.input_proj = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.bn_input = nn.BatchNorm2d(64)
        
        # ST-GCN块
        self.stgcn1 = STGCNBlock(64, 64, kernel_size=9, stride=1)
        self.stgcn2 = STGCNBlock(64, 64, kernel_size=9, stride=1)
        self.stgcn3 = STGCNBlock(64, 128, kernel_size=9, stride=2)
        self.stgcn4 = STGCNBlock(128, 128, kernel_size=9, stride=1)
        self.stgcn5 = STGCNBlock(128, 256, kernel_size=9, stride=2)
        self.stgcn6 = STGCNBlock(256, 256, kernel_size=9, stride=1)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类头
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (batch, time, nodes, channels) -> 需要转换为 (batch, channels, time, nodes)
               或 (batch, time, features) -> 需要reshape
        Returns:
            logits: (batch, num_classes)
        """
        # 处理不同的输入格式
        if len(x.shape) == 3:
            # (batch, time, features) -> (batch, time, nodes, channels)
            # features应该是 nodes * channels
            batch, time, features = x.shape
            if features == self.num_nodes * self.in_channels:
                x = x.view(batch, time, self.num_nodes, self.in_channels)
            else:
                # 假设是扁平化的关键点序列
                x = x.view(batch, time, self.num_nodes, self.in_channels)
        
        # 转换为 (batch, channels, time, nodes)
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2)  # (batch, channels, time, nodes)
        
        # 输入投影
        x = self.input_proj(x)
        x = self.bn_input(x)
        x = F.relu(x)
        
        # ST-GCN块
        x = self.stgcn1(x)
        x = self.stgcn2(x)
        x = self.stgcn3(x)
        x = self.stgcn4(x)
        x = self.stgcn5(x)
        x = self.stgcn6(x)
        
        # 全局池化
        x = self.global_pool(x)  # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 256)
        
        # 分类
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

