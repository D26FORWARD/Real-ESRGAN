import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt # 与RealESRGANModel相同的噪声添加函数
from basicsr.data.transforms import paired_random_crop # 与RealESRGANModel相同的成对裁剪函数
from basicsr.models.sr_model import SRModel # 注意：继承自SRModel，而非SRGANModel
from basicsr.utils import DiffJPEG, USMSharp # 与RealESRGANModel相同的JPEG和USM工具
from basicsr.utils.img_process_util import filter2D # 与RealESRGANModel相同的2D滤波函数
from basicsr.utils.registry import MODEL_REGISTRY # 模型注册表
from torch.nn import functional as F # PyTorch函数式接口


@MODEL_REGISTRY.register() # 将此类注册到MODEL_REGISTRY
class RealESRNetModel(SRModel): # 继承自SRModel
    """RealESRNet模型，用于Real-ESRGAN项目。 (RealESRNet是RealESRGAN项目中不使用GAN的版本)
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    此模型不使用GAN损失进行训练。(Docstring中第二点关于GAN训练的描述似乎是复制粘贴错误)
    主要执行:
    1. 在GPU张量上随机合成LQ图像 (如果使用RealESRGANDataset且启用high_order_degradation)
    2. 使用像素级损失 (如L1) 优化网络 (net_g)。
    """

    def __init__(self, opt):
        super(RealESRNetModel, self).__init__(opt) # 调用父类SRModel的构造函数
        # 以下工具与RealESRGANModel中的相同，用于数据退化合成
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()
        self.queue_size = opt.get('queue_size', 180) # 退化队列大小

    @torch.no_grad() # 此方法不计算梯度
    def _dequeue_and_enqueue(self):
        """训练对池，用于增加批次内的（合成退化）多样性。
        此方法与RealESRGANModel中的完全相同。
        """
        # 初始化队列 (如果尚未存在)
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'队列大小 {self.queue_size} 应该能被批大小 {b} 整除'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0

        if self.queue_ptr == self.queue_size:  # 如果队列已满
            # 执行出队和入队操作
            idx = torch.randperm(self.queue_size) # 随机打乱
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone() # 出队
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            self.queue_lr[0:b, :, :, :] = self.lq.clone() # 当前lq入队首
            self.queue_gt[0:b, :, :, :] = self.gt.clone() # 当前gt入队首
            self.lq = lq_dequeue # 使用出队的lq
            self.gt = gt_dequeue # 使用出队的gt
        else:
            # 队列未满，仅执行入队操作
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad() # 此方法不计算梯度
    def feed_data(self, data):
        """接收来自数据加载器的数据，如果配置了高阶退化，则执行合成LQ图像。
        此方法与RealESRGANModel中的feed_data几乎完全相同，除了处理self.gt_usm的方式略有不同。
        """
        if self.is_train and self.opt.get('high_order_degradation', True):
            # 训练数据合成流程
            self.gt = data['gt'].to(self.device) # 获取GT图像
            # 如果配置了gt_usm，则直接对self.gt进行USM锐化并覆盖self.gt
            # SRModel通常只使用self.gt作为目标，不像SRGANModel那样可能为不同损失使用不同gt (gt, gt_usm)
            if self.opt.get('gt_usm', True): # 检查配置中是否有gt_usm并默认为True
                self.gt = self.usm_sharpener(self.gt)

            # 获取预生成的模糊核
            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4] # GT原始尺寸

            # --- 第一次退化过程 (与RealESRGANModel相同) ---
            out = filter2D(self.gt, self.kernel1) # 注意：这里用self.gt (可能已锐化)
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up': scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down': scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else: scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range'], gray_prob=gray_noise_prob, clip=True, rounds=False)
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

            # --- 第二次退化过程 (与RealESRGANModel相同) ---
            if np.random.uniform() < self.opt['second_blur_prob']: out = filter2D(out, self.kernel2)
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up': scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down': scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else: scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range2'], gray_prob=gray_noise_prob, clip=True, rounds=False)

            # --- 最后处理: JPEG压缩 + Sinc滤波器 + 缩放到最终LQ尺寸 (与RealESRGANModel相同) ---
            if np.random.uniform() < 0.5:
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1); out = self.jpeger(out, quality=jpeg_p)
            else:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1); out = self.jpeger(out, quality=jpeg_p)
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255. # 最终LQ图像

            # 成对随机裁剪 (注意：这里self.gt可能已经是锐化后的版本)
            gt_size = self.opt['gt_size']
            self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['scale'])

            self._dequeue_and_enqueue() # 使用训练对池
            self.lq = self.lq.contiguous()
        else:
            # 用于成对数据训练 (如使用RealESRGANPairedDataset) 或验证
            self.lq = data['lq'].to(self.device) # 直接使用提供的LQ
            if 'gt' in data:
                self.gt = data['gt'].to(self.device) # 直接使用提供的GT
                # 在验证时，如果opt['gt_usm']为True，也会创建一个self.gt_usm变量，但这主要用于可能的评估脚本
                # SRModel的默认test/validation通常只用self.gt进行比较
                if self.opt.get('gt_usm', True): # 检查配置
                    self.gt_usm = self.usm_sharpener(self.gt)


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # 在验证期间不使用合成退化过程
        self.is_train = False # 确保feed_data走 'else' 分支
        # 调用父类SRModel的nondist_validation方法。
        # SRModel.nondist_validation会进行self.net_g(self.lq)并计算指标(如PSNR, SSIM)
        super(RealESRNetModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True # 恢复训练状态

    # optimize_parameters 方法没有在此类中被覆盖(override)。
    # 因此，它将使用父类 SRModel 中的 optimize_parameters 方法。
    # SRModel.optimize_parameters 通常执行以下操作:
    #   1. self.optimizer_g.zero_grad()
    #   2. self.output = self.net_g(self.lq)
    #   3. l_total = self.cri_pix(self.output, self.gt) (或其他在opt中定义的损失)
    #   4. l_total.backward()
    #   5. self.optimizer_g.step()
    # 它只优化生成器 net_g，并且通常只使用像素级损失 (如L1Loss)。
    # 不涉及判别器或GAN损失。
