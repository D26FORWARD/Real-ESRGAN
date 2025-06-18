import cv2  # OpenCV库，用于图像处理
import math # 数学函数
import numpy as np # NumPy库，用于数值计算
import os # 操作系统接口
import os.path as osp # 路径操作模块的别名
import random # 随机数生成
import time # 时间相关函数
import torch # PyTorch深度学习框架
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels # 从basicsr导入退化函数：圆形低通核和随机混合核
from basicsr.data.transforms import augment # 从basicsr导入数据增强函数
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor # 从basicsr导入工具：文件客户端、日志记录器、从字节解码图像、图像转张量
from basicsr.utils.registry import DATASET_REGISTRY # 从basicsr导入数据集注册表
from torch.utils import data as data # PyTorch的数据加载工具


@DATASET_REGISTRY.register() # 装饰器，将此类注册到DATASET_REGISTRY，使其能被框架通过名称调用
class RealESRGANDataset(data.Dataset):
    """用于Real-ESRGAN模型的数据集:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data. (论文名称)

    它加载gt (Ground-Truth, 真实高质量) 图像，并对它们进行增强。
    它还生成模糊核和sinc核，用于生成低质量图像。
    注意：低质量图像是在GPU上以张量形式处理的，以加快处理速度。(本文件只准备GT和退化所需的核参数)

    参数:
        opt (dict): 训练数据集的配置字典。它包含以下键:
            dataroot_gt (str): gt图像的数据根路径。
            meta_info (str): 元信息文件的路径。
            io_backend (dict): IO后端类型和其他参数。
            use_hflip (bool): 是否使用水平翻转。
            use_rot (bool): 是否使用旋转 (通过垂直翻转和转置h、w实现)。
            更多选项请参见代码。
    """

    def __init__(self, opt):
        super(RealESRGANDataset, self).__init__() # 调用父类data.Dataset的构造函数
        self.opt = opt # 保存配置选项
        self.file_client = None # 文件客户端，延迟初始化
        self.io_backend_opt = opt['io_backend'] # IO后端配置
        self.gt_folder = opt['dataroot_gt'] # GT图像文件夹路径

        # 文件客户端 (例如lmdb io后端)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder] # lmdb数据库路径
            self.io_backend_opt['client_keys'] = ['gt'] # 客户端键名
            if not self.gt_folder.endswith('.lmdb'): # 检查路径是否以.lmdb结尾
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            # 从lmdb的meta_info.txt读取文件名列表 (不含扩展名)
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            # 磁盘后端，使用meta_info文件
            # meta_info文件中的每一行描述了一个图像的相对路径
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip().split(' ')[0] for line in fin] # 读取相对路径
                self.paths = [os.path.join(self.gt_folder, v) for v in paths] # 组合成绝对路径

        # 第一次退化的模糊设置
        self.blur_kernel_size = opt['blur_kernel_size'] # 模糊核大小
        self.kernel_list = opt['kernel_list'] # 核类型列表 (如 'iso', 'aniso', 'generalized_iso', 'plateau_iso等')
        self.kernel_prob = opt['kernel_prob']  # 每种核的概率列表
        self.blur_sigma = opt['blur_sigma'] # 模糊标准差范围 [min, max]
        self.betag_range = opt['betag_range']  #广义高斯模糊核的beta_g参数范围
        self.betap_range = opt['betap_range']  # plateau形状模糊核的beta_p参数范围
        self.sinc_prob = opt['sinc_prob']  # 使用sinc滤波器的概率

        # 第二次退化的模糊设置 (参数与第一次类似，但有独立配置)
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # 最终的sinc滤波器概率
        self.final_sinc_prob = opt['final_sinc_prob']

        # 核大小范围，从7到21 (步长为2，即奇数核)
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        # TODO: kernel_range目前是硬编码的，应该放到配置文件中
        #脈衝響應張量 (单位脉冲核)，卷积它不会产生模糊效果，用于当sinc不被选时作为占位符
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1 # 中心点为1，其余为0，构成一个21x21的单位脉冲

    def __getitem__(self, index):
        # 如果文件客户端未初始化，则进行初始化
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- 加载gt图像 -------------------------------- #
        # 形状: (h, w, c); 通道顺序: BGR; 图像范围: [0, 1], float32.
        gt_path = self.paths[index] # 获取当前索引的图像路径
        # 避免因读取文件时高延迟导致的错误，尝试3次
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt') # 通过文件客户端获取图像字节流
            except (IOError, OSError) as e: # 捕获IO或OS错误
                logger = get_root_logger() # 获取日志记录器
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # 更换另一个文件进行读取
                index = random.randint(0, self.__len__() - 1) # 随机选择一个新的索引
                gt_path = self.paths[index]
                time.sleep(1)  # 休眠1秒，应对偶尔的服务器拥堵
            else:
                break # 读取成功则跳出循环
            finally:
                retry -= 1 # 减少尝试次数
        img_gt = imfrombytes(img_bytes, float32=True) # 从字节流解码图像，转换为float32类型

        # -------------------- 训练时数据增强: 翻转, 旋转 -------------------- #
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # 裁剪或填充到400x400 (或者配置文件中指定的大小，但这里硬编码了400)
        # TODO: 400是硬编码的。你可以相应地修改它
        h, w = img_gt.shape[0:2] # 获取图像高宽
        crop_pad_size = 400 # 目标尺寸
        # 填充
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h) # 计算需要填充的高度
            pad_w = max(0, crop_pad_size - w) # 计算需要填充的宽度
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101) # 使用反射填充
        # 裁剪
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # 随机选择裁剪区域的左上角坐标
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...] # 执行裁剪

        # ------------------------ 生成模糊核 (用于第一次退化) ------------------------ #
        kernel_size = random.choice(self.kernel_range) # 从预定义的核大小范围中随机选择一个
        if np.random.uniform() < self.opt['sinc_prob']: # 根据概率决定是否使用sinc核
            # 这个sinc滤波器设置是针对[7, 21]范围的核
            if kernel_size < 13: # 根据核大小调整sinc滤波器的截止频率omega_c
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False) # 生成圆形低通sinc核
        else:
            # 生成混合模糊核 (高斯、广义高斯、plateau等)
            kernel = random_mixed_kernels(
                self.kernel_list, # 核类型列表
                self.kernel_prob, # 各种核的概率
                kernel_size, # 选定的核大小
                self.blur_sigma, # sigma范围
                self.blur_sigma, # sigma范围 (再次传入，可能是历史原因或特定参数需求)
                [-math.pi, math.pi], # 各向异性核的角度范围
                self.betag_range, # 广义高斯beta_g范围
                self.betap_range, # plateau形状beta_p范围
                noise_range=None) # 噪声范围 (此处未使用)
        # 填充核，使其大小统一为21x21 (与pulse_tensor一致)
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ 生成模糊核 (用于第二次退化) ------------------------ #
        #逻辑与第一次退化核的生成类似，但使用第二组配置参数 (kernel_list2, sinc_prob2等)
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']: # 使用sinc_prob2
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2, # 使用kernel_list2
                self.kernel_prob2, # 使用kernel_prob2
                kernel_size,
                self.blur_sigma2, # 使用blur_sigma2
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2, # 使用betag_range2
                self.betap_range2, # 使用betap_range2
                noise_range=None)
        # 填充核，使其大小统一为21x21
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- 最终的sinc核 ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']: # 根据概率决定是否使用最终的sinc核
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21) # 生成sinc核并填充到21x21
            sinc_kernel = torch.FloatTensor(sinc_kernel) # 转换为PyTorch张量
        else:
            sinc_kernel = self.pulse_tensor # 如果不使用sinc核，则使用单位脉冲核 (不产生效果)

        # BGR转RGB, HWC转CHW, numpy转tensor
        # img2tensor函数会处理这些转换，包括通道顺序调整和维度重排
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0] # 将GT图像转换为张量
        kernel = torch.FloatTensor(kernel) # 将第一个模糊核转换为张量
        kernel2 = torch.FloatTensor(kernel2) # 将第二个模糊核转换为张量

        # 返回一个包含GT图像、各种核以及GT路径的字典
        # 注意：这里没有生成LQ图像，LQ图像的生成将在模型训练的主循环中，在GPU上利用这些核进行。
        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path}
        return return_d

    def __len__(self):
        # 返回数据集中图像的总数
        return len(self.paths)
