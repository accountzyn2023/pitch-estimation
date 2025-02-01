import torch
import torch.nn as nn

class PitchDetectionModel(nn.Module):
    """音高检测模型"""
    
    def __init__(self, capacity_multiplier=16):
        """
        初始化音高检测模型
        
        参数:
            capacity_multiplier (int): 容量乘数，用于控制模型大小
                4: tiny
                8: small
                16: medium (默认)
                24: large
                32: full
        """
        super().__init__()
        
        # 基础滤波器数量
        base_filters = [32, 4, 4, 4, 8, 16]
        filters = [f * capacity_multiplier for f in base_filters]
        
        # 构建卷积层
        self.conv_layers = nn.ModuleList()
        
        # 第一个卷积层
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv2d(1, filters[0], kernel_size=(512, 1), stride=(4, 1), padding=(254, 0)),
                nn.ReLU(),
                nn.BatchNorm2d(filters[0]),
                nn.Dropout(0.25),
                nn.MaxPool2d(kernel_size=(2, 1))
            )
        )
        
        # 后续卷积层
        for i in range(1, len(filters)):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(filters[i-1], filters[i], kernel_size=(64, 1), stride=(1, 1), padding=(32, 0)),
                    nn.ReLU(),
                    nn.BatchNorm2d(filters[i]),
                    nn.Dropout(0.25),
                    nn.MaxPool2d(kernel_size=(2, 1))
                )
            )
        
        # 计算展平后的特征维度
        self._calculate_flatten_size()
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 360),
            nn.Sigmoid()
        )
    
    def _calculate_flatten_size(self):
        """计算展平后的特征维度"""
        # 创建更大的示例输入，使用batch_size=2以满足BatchNorm的要求
        x = torch.randn(1, 1, 1024, 1)
        
        # 临时将模型设置为评估模式
        self.eval()
        
        # 通过所有卷积层
        with torch.no_grad():
            for conv_layer in self.conv_layers:
                x = conv_layer(x)
            
            # 转置特征图
            x = x.permute(0, 2, 1, 3)
        
        # 保存展平后的维度（使用单个样本的大小）
        self.flatten_size = x.reshape(1, -1).size(1)
        
        # 恢复为训练模式
        self.train()
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, channels, time_steps, height]
            
        返回:
            torch.Tensor: 音高预测，形状为 [batch_size, 360]
        """
        # 检查输入形状
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D tensor of shape {x.shape}")
        
        # 通过所有卷积层
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
        
        # 转置特征图
        x = x.permute(0, 2, 1, 3)
        
        # 展平
        x = x.reshape(x.size(0), -1)
        
        # 通过全连接层
        x = self.fc(x)
        
        return x

def get_model(model_size='medium'):
    """
    获取指定大小的模型实例
    
    参数:
        model_size (str): 模型大小，可选值：'tiny', 'small', 'medium', 'large', 'full'
        
    返回:
        PitchDetectionModel: 模型实例
    """
    size_multipliers = {
        'tiny': 4,
        'small': 8,
        'medium': 16,
        'large': 24,
        'full': 32
    }
    
    if model_size not in size_multipliers:
        raise ValueError(f"不支持的模型大小: {model_size}。可选值: {list(size_multipliers.keys())}")
    
    return PitchDetectionModel(capacity_multiplier=size_multipliers[model_size])

# 测试代码
if __name__ == "__main__":
    # 创建模型实例
    model = get_model('medium')
    
    # 创建示例输入
    batch_size = 2  # 改小batch_size以便于观察
    x = torch.randn(batch_size, 1, 1024)
    
    # 前向传播
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters())}") 