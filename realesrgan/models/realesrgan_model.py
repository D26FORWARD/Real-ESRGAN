import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt # PyTorch张量级别的高斯和泊松噪声添加函数
from basicsr.data.transforms import paired_random_crop # 成对随机裁剪函数
from basicsr.models.srgan_model import SRGANModel # 基础的SRGAN模型，RealESRGANModel继承自它
from basicsr.utils import DiffJPEG, USMSharp # 可微分JPEG压缩和USM锐化工具
from basicsr.utils.img_process_util import filter2D # 2D滤波函数 (用于模糊)
from basicsr.utils.registry import MODEL_REGISTRY # 模型注册表
from collections import OrderedDict # 有序字典，用于记录损失
from torch.nn import functional as F # PyTorch的函数式接口，如插值


@MODEL_REGISTRY.register() # 将此类注册到MODEL_REGISTRY
class RealESRGANModel(SRGANModel):
    """RealESRGAN模型，源自论文: Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    主要执行:
    1. 在GPU张量上随机合成LQ图像 (核心特点)
    2. 使用GAN训练优化网络
    """

    def __init__(self, opt):
        super(RealESRGANModel, self).__init__(opt) # 调用父类SRGANModel的构造函数
        # 初始化JPEG压缩器，differentiable=False表示在训练生成器时不对此操作求导
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        # 初始化USM锐化器 (Unsharp Masking)
        self.usm_sharpener = USMSharp().cuda()
        # 用于存储近期生成的LQ-GT对的队列大小，以增加批次内退化多样性
        self.queue_size = opt.get('queue_size', 180)

    @torch.no_grad() # 此方法不计算梯度
    def _dequeue_and_enqueue(self):
        """训练对池，用于增加批次内的多样性。

        批处理限制了批次中合成退化的多样性。例如，批次中的样本可能无法具有不同的调整大小缩放因子。
        因此，我们采用此训练对池来增加批次中退化的多样性。
        """
        # 初始化队列 (如果尚未存在)
        b, c, h, w = self.lq.size() # 当前批次的LQ图像尺寸
        if not hasattr(self, 'queue_lr'): # queue_lr是历史LQ图像队列
            assert self.queue_size % b == 0, f'队列大小 {self.queue_size} 应该能被批大小 {b} 整除'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size() # 当前批次的GT图像尺寸 (可能因paired_random_crop而与lq的h,w不同)
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda() # queue_gt是历史GT图像队列
            self.queue_ptr = 0 # 指向队列中下一个要填充的位置的指针

        if self.queue_ptr == self.queue_size:  # 如果队列已满
            # 执行出队和入队操作
            # 随机打乱队列中的现有数据
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # 取出队列头部的b个样本 (出队)
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # 将当前批次的lq和gt样本放入队列头部 (入队)
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()
            # 将出队的样本作为当前批次的lq和gt (即用历史样本替换当前样本)
            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # 队列未满，仅执行入队操作
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b # 更新指针

    @torch.no_grad() # 此方法不计算梯度
    def feed_data(self, data):
        """接收来自数据加载器(dataloader)的数据，然后添加高阶退化以获得LQ图像。
        """
        # is_train 表示当前是否为训练阶段
        # opt.get('high_order_degradation', True) 检查是否启用高阶退化，默认为True
        if self.is_train and self.opt.get('high_order_degradation', True):
            # 训练数据合成流程 (核心)
            self.gt = data['gt'].to(self.device) # 从dataloader获取GT图像，并转移到指定设备 (GPU)
            # 对GT图像进行USM锐化，锐化后的GT通常作为感知损失和GAN损失的目标
            self.gt_usm = self.usm_sharpener(self.gt)

            # 从dataloader获取预先生成的模糊核和sinc核
            self.kernel1 = data['kernel1'].to(self.device) # 第一次模糊的核
            self.kernel2 = data['kernel2'].to(self.device) # 第二次模糊的核
            self.sinc_kernel = data['sinc_kernel'].to(self.device) # Sinc滤波器核

            ori_h, ori_w = self.gt.size()[2:4] # 获取GT图像的原始高宽

            # ----------------------- 第一次退化过程 ----------------------- #
            # 1. 模糊
            out = filter2D(self.gt_usm, self.kernel1) # 使用kernel1对锐化后的GT进行模糊
            # 2. 随机缩放 (resize)
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0] # 按概率选择放大、缩小或保持
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1]) # 在[1, max_scale]范围内随机选择放大倍数
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1) # 在[min_scale, 1]范围内随机选择缩小倍数
            else:
                scale = 1 # 保持原始尺寸
            mode = random.choice(['area', 'bilinear', 'bicubic']) # 随机选择插值方法
            out = F.interpolate(out, scale_factor=scale, mode=mode) # 执行缩放
            # 3. 添加噪声
            gray_noise_prob = self.opt['gray_noise_prob'] # 灰度噪声的概率
            if np.random.uniform() < self.opt['gaussian_noise_prob']: # 按概率添加高斯噪声
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else: # 否则添加泊松噪声
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # 4. JPEG压缩
            # 从配置的jpeg_range中为批内每张图片随机采样一个JPEG质量因子
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # JPEGer前确保图像值在[0,1]范围，否则可能产生不期望的伪影
            out = self.jpeger(out, quality=jpeg_p) # 执行JPEG压缩

            # ----------------------- 第二次退化过程 ----------------------- #
            # 1. 模糊 (可选)
            if np.random.uniform() < self.opt['second_blur_prob']: # 按概率决定是否进行第二次模糊
                out = filter2D(out, self.kernel2) # 使用kernel2进行模糊
            # 2. 随机缩放 (目标是生成最终LQ尺寸的中间步骤)
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            # 计算目标尺寸：(原始GT高度 / 最终SR缩放比例 * 当前随机缩放因子, ...)
            # 这是为了使最终LQ图像的尺寸是GT图像的 1/self.opt['scale']
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            # 3. 添加噪声
            gray_noise_prob = self.opt['gray_noise_prob2'] # 第二轮噪声的灰度概率
            if np.random.uniform() < self.opt['gaussian_noise_prob2']: # 第二轮高斯噪声概率
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else: # 第二轮泊松噪声
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # 最后一步：JPEG压缩 + Sinc滤波器 (以及缩放到最终LQ尺寸)
            # 需要将图像缩放到期望的尺寸。我们将[缩放回目标尺寸 + Sinc滤波器]组合为一个操作。
            # 考虑两种顺序:
            #   1. [缩放回目标尺寸 + Sinc滤波器] + JPEG压缩
            #   2. JPEG压缩 + [缩放回目标尺寸 + Sinc滤波器]
            # 经验发现其他组合 (例如 Sinc + JPEG + Resize) 会引入扭曲的线条。
            if np.random.uniform() < 0.5: # 随机选择以上两种顺序之一
                # 顺序1: 先缩放和Sinc滤波，再JPEG
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                # 缩放到最终的LQ尺寸 (GT尺寸 / SR缩放比例)
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel) # 应用Sinc滤波器
                # JPEG压缩
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2']) # 第二轮JPEG质量范围
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # 顺序2: 先JPEG，再缩放和Sinc滤波
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # 缩放和Sinc滤波
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # 将图像值限制在[0,255]并四舍五入，然后归一化到[0,1]，作为最终的LQ图像
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # 对生成的LQ图像和原始GT图像 (及锐化版GT) 进行成对随机裁剪
            # 这是为了确保训练时网络看到的LQ和GT在空间上是对应的patch
            gt_size = self.opt['gt_size'] # GT patch的尺寸
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            # 将当前生成的LQ-GT对加入到训练对池中
            self._dequeue_and_enqueue()
            # 再次锐化self.gt，因为_dequeue_and_enqueue可能已经用池中旧的gt替换了当前的self.gt
            self.gt_usm = self.usm_sharpener(self.gt)
            # contiguous()确保张量在内存中是连续的，有时是某些操作或避免警告所必需的
            self.lq = self.lq.contiguous()
        else:
            # 用于成对数据训练 (如使用RealESRGANPairedDataset) 或验证阶段
            # 直接从dataloader获取LQ图像
            self.lq = data['lq'].to(self.device)
            if 'gt' in data: # 如果dataloader提供了GT图像 (例如验证时)
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt) # 也对验证GT进行锐化，可能用于评估

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # 在验证期间不使用合成退化过程
        self.is_train = False # 设置为False，feed_data将走else分支，直接使用dataloader提供的lq, gt
        super(RealESRGANModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True # 验证结束后恢复为训练状态

    def optimize_parameters(self, current_iter):
        # 根据配置决定L1损失、感知损失和GAN损失的GT目标是原始GT还是USM锐化后的GT
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False: # 如果l1_gt_usm配置为False，则L1损失使用原始gt
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False: # 如果percep_gt_usm配置为False，则感知损失使用原始gt
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False: # 如果gan_gt_usm配置为False，则GAN损失的真实目标使用原始gt
            gan_gt = self.gt

        # 优化生成器 (net_g)
        # 首先冻结判别器 (net_d) 的参数，不计算其梯度
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad() # 清空生成器的梯度
        self.output = self.net_g(self.lq) # 生成器前向传播，得到超分结果 self.output

        l_g_total = 0 # 初始化生成器总损失
        loss_dict = OrderedDict() # 使用有序字典记录各种损失
        # 根据训练迭代次数决定是否更新判别器 (net_d_iters控制判别器更新频率)
        # net_d_init_iters 表示初始阶段不训练判别器的迭代次数
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # 像素损失 (L1 Loss)
            if self.cri_pix: # 如果定义了像素损失 (cri_pix)
                l_g_pix = self.cri_pix(self.output, l1_gt) # 计算L1损失
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # 感知损失 (Perceptual Loss) 和风格损失 (Style Loss)
            if self.cri_perceptual: # 如果定义了感知损失 (cri_perceptual)
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt) # 计算感知和风格损失
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None: # 通常在SRGAN中不单独使用风格损失，但基础框架支持
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # GAN损失 (对抗损失)
            fake_g_pred = self.net_d(self.output) # 将生成器的输出送入判别器
            # 计算生成器的GAN损失，目标是让判别器认为生成的图像是真实的 (is_disc=False)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward() # 生成器总损失反向传播
            self.optimizer_g.step() # 更新生成器参数

        # 优化判别器 (net_d)
        # 解冻判别器的参数
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad() # 清空判别器的梯度
        # 对真实图像进行判别
        real_d_pred = self.net_d(gan_gt) # 将真实的GT (或其锐化版) 送入判别器
        # 计算真实图像的GAN损失，目标是让判别器识别为真实 (True)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach()) # 记录判别器对真实图像的平均输出
        l_d_real.backward() # 反向传播

        # 对虚假图像 (生成器输出) 进行判别
        # detach()用于阻断梯度流向生成器，clone()是为了兼容PyTorch 1.9+的一个bug
        fake_d_pred = self.net_d(self.output.detach().clone())
        # 计算虚假图像的GAN损失，目标是让判别器识别为虚假 (False)
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach()) # 记录判别器对虚假图像的平均输出
        l_d_fake.backward() # 反向传播
        self.optimizer_d.step() # 更新判别器参数

        # 如果启用了EMA (Exponential Moving Average)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay) # 更新生成器的EMA模型

        # 记录所有损失值，用于日志或TensorBoard显示 (reduce_loss_dict用于分布式训练时聚合损失)
        self.log_dict = self.reduce_loss_dict(loss_dict)
