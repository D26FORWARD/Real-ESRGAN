import cv2 # OpenCV库，用于图像读写和颜色空间转换等
import math # 数学库，用于向上取整 (math.ceil)
import numpy as np # NumPy库，用于数值和数组操作
import os # 操作系统接口，用于路径操作
import queue # 队列库，用于PrefetchReader
import threading # 线程库，用于PrefetchReader和IOConsumer
import torch # PyTorch深度学习框架
from basicsr.utils.download_util import load_file_from_url # 从URL下载文件的工具
from torch.nn import functional as F # PyTorch的函数式接口，如padding

# 获取项目根目录 (realesrgan文件夹的父目录)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RealESRGANer():
    """一个用于通过RealESRGAN放大图像的辅助类。

    参数:
        scale (int): 网络中使用的上采样缩放因子。通常是2或4。
        model_path (str): 预训练模型的路径。可以是URL (会自动先下载)。
        model (nn.Module): 定义好的网络模型实例。默认为None。 (实际使用时必须传入)
        tile (int): 由于过大图像会导致GPU显存不足，此选项会将输入图像首先裁剪成瓦片(tile)，
                    然后分别处理每个瓦片，最后将它们合并成一张图像。0表示不使用瓦片处理。默认为0。
        tile_pad (int): 每个瓦片的填充大小，用于移除边界伪影。默认为10。
        pre_pad (int): 对输入图像进行的预填充，以避免边界伪影。默认为10。
        half (bool): 推理时是否使用半精度 (float16)。默认为False。
        device (torch.device): 指定运行设备。如果为None，则自动选择。
        gpu_id (int): 指定使用的GPU ID。如果device也为None时生效。
    """

    def __init__(self,
                 scale, # 缩放比例
                 model_path, # 模型路径或URL
                 dni_weight=None, # Deep Network Interpolation (DNI) 权重，如果model_path是列表则使用
                 model=None, # 预加载的PyTorch模型
                 tile=0, # 瓦片大小，0表示不分块
                 tile_pad=10, # 瓦片之间的重叠区域
                 pre_pad=10, # 图像预处理时的边缘填充
                 half=False, # 是否使用半精度推理
                 device=None, # 指定设备
                 gpu_id=None): # 指定GPU ID
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None # 用于确保图像尺寸能被特定因子整除的填充参数
        self.half = half

        # 初始化模型运行的设备
        if gpu_id: # 如果指定了gpu_id
            self.device = torch.device(
                f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
        else: # 未指定gpu_id
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        # 加载模型权重
        if isinstance(model_path, list): # 如果model_path是列表，表示使用DNI
            assert model is not None, 'model should be provided when using DNI.'
            assert len(model_path) == len(dni_weight), 'model_path和dni_weight应具有相同的长度。'
            loadnet = self.dni(model_path[0], model_path[1], dni_weight) # 执行DNI
        else: # 单个模型路径
            # 如果model_path以https开头，则先从URL下载模型到 'weights' 文件夹
            if model_path.startswith('https://'):
                model_path = load_file_from_url(
                    url=model_path, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
            loadnet = torch.load(model_path, map_location=torch.device('cpu')) # 加载模型权重到CPU

        # 优先使用 'params_ema' (Exponential Moving Average) 的权重，如果存在的话
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params' # 否则使用常规参数

        if model is None: # 如果外部没有传入实例化模型，则尝试从loadnet中获取（通常不推荐这样做，应由外部实例化网络结构）
            raise ValueError("A model instance (nn.Module) must be provided.")

        model.load_state_dict(loadnet[keyname], strict=True) # 将权重加载到模型中

        model.eval() # 设置模型为评估模式 (关闭dropout, batchnorm更新等)
        self.model = model.to(self.device) # 将模型移到指定设备
        if self.half: # 如果使用半精度
            self.model = self.model.half()

    def dni(self, net_a_path, net_b_path, dni_weight, key='params', loc='cpu'):
        """深度网络插值 (Deep Network Interpolation)。
        论文: "Deep Network Interpolation for Continuous Imagery Effect Transition"
        通过对两个预训练模型的权重进行加权平均，实现模型效果的平滑过渡。

        参数:
            net_a_path (str): 第一个模型的路径。
            net_b_path (str): 第二个模型的路径。
            dni_weight (list[float]): 两个模型的插值权重，例如 [0.5, 0.5]。
            key (str): 权重字典中的键名，通常是 'params' 或 'params_ema'。
            loc (str): 加载模型时的设备位置。
        返回:
            dict: 插值后的模型权重字典 (net_a的权重被修改)。
        """
        net_a_weights = torch.load(net_a_path, map_location=torch.device(loc)) # 加载模型A的权重
        net_b_weights = torch.load(net_b_path, map_location=torch.device(loc)) # 加载模型B的权重
        # 遍历模型A的每一层权重
        for k, v_a in net_a_weights[key].items():
            # 对权重进行加权平均
            net_a_weights[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b_weights[key][k]
        return net_a_weights # 返回修改后的模型A的权重字典

    def pre_process(self, img):
        """预处理图像，例如预填充(pre-pad)和模数填充(mod pad)，以确保图像尺寸可被特定值整除。
        """
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float() # HWC, NumPy -> CHW, Tensor
        self.img = img.unsqueeze(0).to(self.device) # CHW -> BCHW, 并移到设备
        if self.half: # 如果使用半精度
            self.img = self.img.half()

        # 预填充 (pre_pad)，在图像四周填充，以减少边缘伪影
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect') # 使用反射填充

        # 模数填充 (mod pad)，确保图像高宽能被mod_scale整除 (某些网络结构需要)
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1: # 对于 scale 1 (例如 RealESRGAN_x1plus)，可能需要更大的 mod_scale
            self.mod_scale = 4
        # else: self.mod_scale 保持 None，不进行mod_pad

        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0 # 初始化高宽填充量
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0): # 如果高度不能整除
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale) # 计算需要填充的高度
            if (w % self.mod_scale != 0): # 如果宽度不能整除
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale) # 计算需要填充的宽度
            # 在图像右边和下边进行填充
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        # 模型推理 (对整个图像进行)
        self.output = self.model(self.img)

    def tile_process(self):
        """瓦片处理：首先将输入图像裁剪成瓦片，然后逐个处理每个瓦片。
        最后，所有处理后的瓦片被合并成一张图像。
        修改自: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape # 输入图像尺寸
        output_height = height * self.scale # 计算输出图像高度
        output_width = width * self.scale # 计算输出图像宽度
        output_shape = (batch, channel, output_height, output_width)

        # 初始化一个全黑的输出图像
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size) # 计算x方向的瓦片数
        tiles_y = math.ceil(height / self.tile_size) # 计算y方向的瓦片数

        # 遍历所有瓦片
        for y in range(tiles_y):
            for x in range(tiles_x):
                # 提取当前瓦片在输入图像中的位置 (无填充)
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # 计算当前瓦片在输入图像中的位置 (带填充 tile_pad)
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # 当前瓦片的实际尺寸 (无填充)
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y

                tile_idx = y * tiles_x + x + 1 # 当前瓦片索引 (用于打印)
                # 提取带填充的输入瓦片
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # 对瓦片进行上采样
                try:
                    with torch.no_grad(): # 推理时不需要计算梯度
                        output_tile = self.model(input_tile)
                except RuntimeError as error: # 捕获可能的运行时错误 (如OOM)
                    print('错误', error)
                print(f'\t瓦片 {tile_idx}/{tiles_x * tiles_y}')

                # 计算输出瓦片在最终输出图像中的位置 (对应无填充的输入区域)
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # 计算从output_tile中裁剪出有效区域 (去除tile_pad对应放大部分) 的坐标
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # 将处理后的瓦片 (的有效部分) 放入最终输出图像的对应位置
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        # 移除预处理时添加的额外填充
        if self.mod_scale is not None: # 移除mod_pad
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        if self.pre_pad != 0: # 移除pre_pad
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

    @torch.no_grad() # 主增强函数，不计算梯度
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        """对输入的图像(img)进行增强和上采样。

        参数:
            img (np.ndarray): 输入图像，BGR或Grayscale格式的NumPy数组。
            outscale (float): 最终输出的缩放因子。如果与模型的scale不同，则进行额外缩放。默认为None。
            alpha_upsampler (str): alpha通道的上采样方法。'realesrgan'表示用模型放大，否则用线性插值。
        返回:
            tuple: (output_img, img_mode)
                   output_img (np.ndarray): 处理后的图像。
                   img_mode (str): 原始图像的模式 ('L', 'RGB', 'RGBA')。
        """
        h_input, w_input = img.shape[0:2] # 记录原始输入尺寸，用于最终outscale调整
        # img: numpy数组
        img = img.astype(np.float32) # 转换为float32类型
        if np.max(img) > 256:  # 判断是否为16位图像 (像素值大于255)
            max_range = 65535
            print('\t输入是16位图像')
        else:
            max_range = 255
        img = img / max_range # 归一化到[0,1]范围

        img_mode = None # 初始化图像模式
        if len(img.shape) == 2:  # 灰度图像
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # 转换为RGB进行处理
        elif img.shape[2] == 4:  # RGBA图像 (带alpha通道)
            img_mode = 'RGBA'
            alpha = img[:, :, 3] # 分离alpha通道
            img = img[:, :, 0:3] # 取RGB通道
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
            if alpha_upsampler == 'realesrgan': # 如果alpha通道也用模型放大
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB) # alpha通道也转为3通道RGB格式给模型
        else: # RGB图像
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB

        # ------------------- 处理图像 (不含alpha通道) ------------------- #
        self.pre_process(img) # 预处理
        if self.tile_size > 0: # 如果设置了瓦片大小
            self.tile_process() # 进行瓦片处理
        else:
            self.process() # 对整个图像进行处理
        output_img = self.post_process() # 后处理，移除填充

        # 将输出张量转换回NumPy图像 (HWC, BGR格式)
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0)) # RGB -> BGR, CHW -> HWC

        if img_mode == 'L': # 如果原始是灰度图，则转换回灰度
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- 必要时处理alpha通道 ------------------- #
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan': # 如果alpha通道用模型放大
                self.pre_process(alpha) # 对alpha通道进行同样的预处理
                if self.tile_size > 0:
                    self.tile_process() # 瓦片处理
                else:
                    self.process() # 整体处理
                output_alpha = self.post_process() # 后处理
                # 将alpha输出张量转为NumPy灰度图
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # 使用cv2的线性插值放大alpha通道
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            # 合并处理后的RGB和alpha通道
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA) # 先将BGR转为BGRA
            output_img[:, :, 3] = output_alpha # 赋值alpha通道

        # ------------------------------ 返回结果 ------------------------------ #
        # 根据原始图像位深，将像素值反归一化到[0, 255]或[0, 65535]
        if max_range == 65535:  # 16-bit
            output = (output_img * 65535.0).round().astype(np.uint16)
        else: # 8-bit
            output = (output_img * 255.0).round().astype(np.uint8)

        # 如果指定了outscale且与模型scale不同，则进行最终尺寸调整
        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (
                    int(w_input * outscale), # 基于原始输入尺寸和目标outscale计算最终宽高
                    int(h_input * outscale),
                ), interpolation=cv2.INTER_LANCZOS4) # 使用Lanczos4插值

        return output, img_mode


class PrefetchReader(threading.Thread):
    """预取图像的线程类。

    参数:
        img_list (list[str]): 待读取图像路径的列表。
        num_prefetch_queue (int): 预取队列的大小。
    """

    def __init__(self, img_list, num_prefetch_queue):
        super().__init__() # 调用父类threading.Thread的构造函数
        self.que = queue.Queue(num_prefetch_queue) # 初始化队列
        self.img_list = img_list

    def run(self):
        # 线程执行体：遍历图像列表，读取图像并放入队列
        for img_path in self.img_list:
            try:
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # 读取图像，保持原始通道数
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
                img = None # 或者进行其他错误处理
            self.que.put(img)

        self.que.put(None) # 所有图像读取完毕后，放入None作为结束标志

    def __next__(self):
        # 迭代器协议：获取队列中的下一项
        next_item = self.que.get()
        if next_item is None: # 如果是结束标志
            raise StopIteration # 停止迭代
        return next_item

    def __iter__(self):
        # 迭代器协议：返回自身
        return self


class IOConsumer(threading.Thread):
    """图像IO消费者线程类，用于将处理后的图像写入磁盘。

    参数:
        opt (dict): 配置选项 (可能包含保存路径等信息，但在此run方法中未直接使用opt)。
        que (queue.Queue): 存储待保存图像信息的队列。
        qid (int): 消费者线程的ID (用于打印日志)。
    """
    def __init__(self, opt, que, qid):
        super().__init__()
        self._queue = que
        self.qid = qid
        self.opt = opt # 保存配置，但当前run方法未用到

    def run(self):
        # 线程执行体：不断从队列中获取消息并处理
        while True:
            msg = self._queue.get() # 从队列获取消息
            if isinstance(msg, str) and msg == 'quit': # 如果是退出消息
                break # 结束线程

            # 假设消息是一个字典，包含输出图像和保存路径
            output = msg['output']
            save_path = msg['save_path']
            try:
                cv2.imwrite(save_path, output) # 保存图像
            except Exception as e:
                print(f"Error writing image {save_path}: {e}")
                # 可选: 进行错误处理，例如记录失败的路径

        print(f'IO消费者线程 {self.qid} 已完成。')
