from basicsr.utils.registry import ARCH_REGISTRY  # 从 basicsr 框架导入架构注册器
from torch import nn as nn  # 导入 PyTorch 的神经网络模块，并简写为 nn
from torch.nn import functional as F  # 导入 PyTorch 神经网络模块中的函数库，并简写为 F


@ARCH_REGISTRY.register()  # 装饰器，将 SRVGGNetCompact 类注册到 ARCH_REGISTRY
class SRVGGNetCompact(nn.Module):
    """一个用于超分辨率的紧凑型VGG风格网络结构。

    这是一个紧凑的网络结构，它在最后一层执行上采样，并且不在高分辨率特征空间上进行卷积。
    这种设计可以减少计算量，使得网络更轻量。

    参数:
        num_in_ch (int): 输入通道数。默认为 3 (RGB图像)。
        num_out_ch (int): 输出通道数。默认为 3 (RGB图像)。
        num_feat (int): 中间特征的通道数。默认为 64。
        num_conv (int): 主体网络中卷积层的数量。默认为 16。
        upscale (int): 上采样因子。默认为 4。
        act_type (str): 激活函数类型，可选: 'relu', 'prelu', 'leakyrelu'。默认为 'prelu'。
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()  # 调用父类 nn.Module 的构造函数
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()  # 使用 ModuleList 来存储网络主体部分的层序列

        # 第一个卷积层：将输入通道映射到 num_feat 通道
        self.body.append(nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1))

        # 第一个激活层
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            # PReLU 允许每个通道学习一个单独的负斜率参数
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            raise ValueError(f"Unsupported activation type: {act_type}")
        self.body.append(activation)

        # 主体结构：包含 num_conv 个卷积层和对应的激活层
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1))
            # 激活层
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            # else: # 此处已在上层检查过，可省略
            #     raise ValueError(f"Unsupported activation type: {act_type}")
            self.body.append(activation)

        # 最后一个卷积层：准备用于 PixelShuffle 上采样的特征图
        # 输出通道数为 num_out_ch * upscale^2，这是 PixelShuffle 的要求
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, kernel_size=3, stride=1, padding=1))

        # 上采样层：使用 PixelShuffle 进行高效的上采样
        # PixelShuffle 将形状为 (N, C * r^2, H, W) 的张量重塑为 (N, C, H * r, W * r)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        # 输入 x: (N, num_in_ch, H_in, W_in)
        out = x  # 将输入赋值给 out

        # 依次通过主体网络中的所有层
        for layer in self.body: # 更 Pythonic 的遍历方式
            out = layer(out)
        # 经过 body 后，out 的形状为 (N, num_out_ch * upscale^2, H_in, W_in)

        # 通过 PixelShuffle 进行上采样
        out = self.upsampler(out) # out 的形状变为 (N, num_out_ch, H_in * upscale, W_in * upscale)

        # 添加最近邻上采样的原始图像，使得网络学习残差
        # F.interpolate 用于对原始输入 x 进行简单的上采样，作为基准 (base)
        # mode='nearest' 是一种快速且简单的上采样方法
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        # 将网络输出与基准上采样图像相加，形成残差学习
        out += base # (N, num_out_ch, H_out, W_out)

        return out
