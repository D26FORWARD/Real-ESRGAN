from basicsr.utils.registry import ARCH_REGISTRY  # 从 basicsr 框架导入架构注册器，用于注册此判别器模型
from torch import nn as nn  # 导入 PyTorch 的神经网络模块，并简写为 nn
from torch.nn import functional as F  # 导入 PyTorch 神经网络模块中的函数库，并简写为 F
from torch.nn.utils import spectral_norm  # 从 PyTorch 的工具中导入谱归一化函数


@ARCH_REGISTRY.register()  # 装饰器，将 UNetDiscriminatorSN 类注册到 ARCH_REGISTRY，使其能被框架通过名称调用
class UNetDiscriminatorSN(nn.Module):
    """定义一个带有谱归一化 (SN) 的 U-Net 判别器。

    该判别器用于 Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data 论文中。

    参数:
        num_in_ch (int): 输入图像的通道数。默认为 3 (RGB图像)。
        num_feat (int): 基础中间特征的通道数。默认为 64。
        skip_connection (bool): 是否在 U-Net 中使用跳跃连接。默认为 True。
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()  # 调用父类 nn.Module 的构造函数
        self.skip_connection = skip_connection  # 是否使用跳跃连接
        norm = spectral_norm  # 将谱归一化函数赋值给局部变量 norm，方便后续使用

        # 第一个卷积层，不使用谱归一化，将输入通道数映射到 num_feat
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        # 下采样模块 (编码器部分)
        # 每一层卷积后通道数翻倍，图像尺寸减半 (stride=2)
        # 使用谱归一化 (norm)，bias 设置为 False 是因为谱归一化本身可以起到类似偏置的作用或后续有归一化层
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, kernel_size=4, stride=2, padding=1, bias=False))

        # 上采样模块 (解码器部分)
        # 每一层卷积后通道数减半
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, stride=1, padding=1, bias=False))

        # 额外的卷积层，用于进一步处理特征
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=False))
        # 最后一个卷积层，输出单通道的判别结果 (raw score)，不使用谱归一化和激活函数
        self.conv9 = nn.Conv2d(num_feat, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 输入 x: (N, num_in_ch, H, W)

        # 第一个卷积层 + LeakyReLU 激活
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)  # (N, num_feat, H, W)

        # 下采样
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True) # (N, num_feat*2, H/2, W/2)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True) # (N, num_feat*4, H/4, W/4)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True) # (N, num_feat*8, H/8, W/8)

        # 上采样 + 特征融合 (如果启用跳跃连接)
        # F.interpolate 用于上采样特征图
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False) # (N, num_feat*8, H/4, W/4)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True) # (N, num_feat*4, H/4, W/4)

        if self.skip_connection:
            x4 = x4 + x2  # 跳跃连接：融合来自编码器对应层级的特征 x2

        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False) # (N, num_feat*4, H/2, W/2)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True) # (N, num_feat*2, H/2, W/2)

        if self.skip_connection:
            x5 = x5 + x1  # 跳跃连接：融合来自编码器对应层级的特征 x1

        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False) # (N, num_feat*2, H, W)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True) # (N, num_feat, H, W)

        if self.skip_connection:
            x6 = x6 + x0  # 跳跃连接：融合来自编码器对应层级的特征 x0

        # 额外的卷积层
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True) # (N, num_feat, H, W)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True) # (N, num_feat, H, W)
        # 输出层，得到判别结果，通常是一个二维的 PatchGAN 形式的输出
        out = self.conv9(out)  # (N, 1, H, W)

        return out # 返回判别器的原始输出分数，后续会通过 sigmoid 转换为概率或直接用于计算损失
