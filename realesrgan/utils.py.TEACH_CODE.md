# `realesrgan/utils.py` 代码分析

## 1. 文件概述

`realesrgan/utils.py` 文件包含 Real-ESRGAN 项目中用于推理和批量处理图像的辅助工具。其中最重要的组件是 `RealESRGANer` 类，它封装了使用预训练 Real-ESRGAN 模型进行图像上采样的完整流程，包括模型加载、预处理、瓦片式推理（tile processing）以处理大图像、后处理以及对alpha通道的处理。此外，该文件还定义了 `PrefetchReader` 和 `IOConsumer` 两个基于线程的类，用于在批量推理脚本中优化图像的读取和保存，提高IO效率。

## 2. `RealESRGANer` 类详解

`RealESRGANer` 是执行实际图像放大操作的核心类，通常在 `inference_realesrgan.py` 等推理脚本中使用。

### 2.1. 构造函数 `__init__`

```python
class RealESRGANer():
    def __init__(self,
                 scale, # 模型放大倍数 (例如 2 或 4)
                 model_path, # 模型权重文件路径或URL
                 dni_weight=None, # DNI插值权重 (可选)
                 model=None, # 预实例化的PyTorch模型 (必须提供)
                 tile=0, # 瓦片大小 (0表示不分块)
                 tile_pad=10, # 瓦片填充
                 pre_pad=10, # 图像预填充
                 half=False, # 半精度推理
                 device=None, # 设备 (cuda/cpu)
                 gpu_id=None): # GPU ID
        self.scale = scale
        self.tile_size = tile
        # ... 其他参数保存 ...
        self.device = torch.device(...) # 设置运行设备

        # 模型加载逻辑
        if isinstance(model_path, list): # DNI模式
            assert model is not None, '使用DNI时必须提供模型实例 (model should be provided when using DNI.)'
            assert len(model_path) == len(dni_weight), '模型路径列表和DNI权重列表长度应一致 (model_path and dni_weight should have the same length.)'
            loadnet = self.dni(model_path[0], model_path[1], dni_weight) # 执行DNI插值
        else: # 单模型模式
            if model_path.startswith('https://'): # 如果是URL，则下载
                model_path = load_file_from_url(...)
            loadnet = torch.load(model_path, map_location=torch.device('cpu')) # 加载权重

        if model is None:
             raise ValueError("必须提供模型实例 (A model instance (nn.Module) must be provided.)")

        # 加载权重到模型 (优先使用EMA权重)
        keyname = 'params_ema' if 'params_ema' in loadnet else 'params'
        model.load_state_dict(loadnet[keyname], strict=True)

        model.eval() # 设置为评估模式
        self.model = model.to(self.device) # 模型移至设备
        if self.half: self.model = self.model.half() # 半精度转换
```

*   **参数**:
    *   `scale`: 网络的放大倍数。
    *   `model_path`: 可以是单个模型文件的路径（`.pth`），也可以是一个包含两个模型路径的列表（用于DNI）。如果是URL，会自动下载。
    *   `dni_weight`: 一个包含两个浮点数的列表，表示DNI中两个模型的权重。仅当 `model_path` 是列表时使用。
    *   `model`: **必须传入**一个已经实例化的PyTorch `nn.Module` 对象（例如 `SRVGGNetCompact` 的实例）。`RealESRGANer` 负责将权重加载到这个模型实例中。
    *   `tile`: 瓦片处理的瓦片大小。如果为0，则对整个图像进行处理（可能导致大图像OOM）。
    *   `tile_pad`: 瓦片之间的重叠区域大小，用于减少拼接缝隙处的伪影。
    *   `pre_pad`: 对整个输入图像进行的初始填充，同样为了减少边缘伪影。
    *   `half`: 是否启用FP16半精度推理，可以加速并减少显存占用，但可能轻微影响精度。
    *   `device`/`gpu_id`: 指定运算设备。
*   **模型加载**:
    *   **DNI (Deep Network Interpolation)**: 如果 `model_path` 是一个列表（包含两个模型路径）并且提供了 `dni_weight`，则会调用 `self.dni` 方法对这两个模型的权重进行加权平均，生成一个新的权重集。
    *   **标准加载**: 如果是单个模型路径，则直接加载。如果是URL，则先下载到 `weights` 目录。
    *   **权重选择**: 优先加载名为 `params_ema` 的EMA（Exponential Moving Average）权重，如果不存在，则加载 `params` 权重。EMA权重通常更平滑，泛化性能更好。
    *   将加载的权重载入到传入的 `model` 实例中，设置为评估模式 (`model.eval()`)，并移到指定设备。

### 2.2. 深度网络插值 `dni`

```python
    def dni(self, net_a_path, net_b_path, dni_weight, key='params', loc='cpu'):
        net_a_weights = torch.load(net_a_path, map_location=torch.device(loc))
        net_b_weights = torch.load(net_b_path, map_location=torch.device(loc))
        for k, v_a in net_a_weights[key].items(): # 遍历模型A的参数
            # net_a_param = weight0 * v_a + weight1 * v_b
            net_a_weights[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b_weights[key][k]
        return net_a_weights # 返回被修改后的 net_a_weights
```
*   此方法实现了两个模型权重的线性插值。这允许用户通过调整 `dni_weight` 来混合两个不同模型（例如，一个锐利模型和一个平滑模型）的特性，以达到某种中间效果。

### 2.3. 预处理 `pre_process`

```python
    def pre_process(self, img):
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float() # HWC, NumPy -> CHW, Tensor
        self.img = img.unsqueeze(0).to(self.device) # CHW -> BCHW (batch=1)
        if self.half: self.img = self.img.half()

        # 预填充 (pre_pad)
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect') # 左右和上下对称填充

        # 模数填充 (mod_pad)，确保图像尺寸能被网络特定层整除
        if self.scale == 2: self.mod_scale = 2
        elif self.scale == 1: self.mod_scale = 4 # x1模型可能对尺寸有特殊要求

        if self.mod_scale is not None:
            # ... 计算 self.mod_pad_h, self.mod_pad_w ...
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect') # 右下填充
```
*   将输入的NumPy图像（通常是HWC BGR格式，已在`enhance`中转为RGB）转换为PyTorch张量（BCHW RGB格式），并根据需要转为半精度。
*   `self.pre_pad`: 对图像的四个边缘进行反射填充，目的是减少在网络处理图像边缘时可能产生的伪影。
*   `self.mod_scale`: 某些网络由于其下采样/上采样结构，可能要求输入图像的尺寸是特定数字（如2、4、8）的倍数。此步骤计算需要额外填充的量，以满足该要求。

### 2.4. 推理 (`process` 和 `tile_process`)

*   **`process(self)`**:
    ```python
        self.output = self.model(self.img) # 对整个图像（可能已填充）进行模型前向传播
    ```
    直接对整个预处理后的图像 `self.img` 进行模型推理。适用于图像较小，可以完整放入GPU显存的情况。

*   **`tile_process(self)`**:
    ```python
    # ... (计算瓦片数量 tiles_x, tiles_y) ...
    # self.output = self.img.new_zeros(output_shape) # 初始化空白输出图像
    # for y in range(tiles_y):
    #     for x in range(tiles_x):
    #         # ... (计算当前瓦片的输入坐标 input_start_x/y, input_end_x/y) ...
    #         # ... (计算带重叠区域的瓦片输入坐标 input_start_x/y_pad, input_end_x/y_pad) ...
    #         input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad] # 提取输入瓦片
    #         output_tile = self.model(input_tile) # 对瓦片进行推理
    #         # ... (计算瓦片在输出图像中的对应位置 output_start_x/y, output_end_x/y) ...
    #         # ... (计算从 output_tile 中提取有效区域的坐标 output_start_x/y_tile, output_end_x/y_tile) ...
    #         # 将有效区域结果填回 self.output
    #         self.output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
    #             output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
    ```
    当输入图像过大时，使用此方法。
    1.  将输入图像分割成多个小瓦片（tiles）。
    2.  每个瓦片在提取时会包含额外的重叠区域（`self.tile_pad`）。
    3.  对每个带重叠区域的瓦片独立进行模型推理。
    4.  从处理后的瓦片中，只取其对应原始瓦片（无重叠）的部分，拼接回最终的输出图像。重叠区域有助于减少瓦片拼接处的明显缝隙和伪影。

### 2.5. 后处理 `post_process`

```python
    def post_process(self):
        # 移除 self.mod_pad 和 self.pre_pad 对应的区域
        if self.mod_scale is not None:
            # ... (裁剪掉 self.mod_pad_h * self.scale 和 self.mod_pad_w * self.scale) ...
        if self.pre_pad != 0:
            # ... (裁剪掉 self.pre_pad * self.scale) ...
        return self.output
```
*   移除在 `pre_process` 阶段添加的 `mod_pad` 和 `pre_pad`。注意移除的填充量需要乘以 `self.scale`，因为是在放大后的图像上操作。

### 2.6. 主增强方法 `enhance`

这是用户调用的主要接口。

```python
    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        h_input, w_input = img.shape[0:2] # 保存原始输入尺寸
        # 1. 输入图像格式与归一化 (NumPy HWC BGR -> NumPy HWC RGB [0,1])
        # ... (判断8位/16位, 灰度/RGBA/RGB, 转换为RGB, 归一化到[0,1]) ...
        # ... (分离alpha通道，如果存在) ...

        # 2. 核心处理 (RGB通道)
        self.pre_process(img_rgb) # 预处理
        if self.tile_size > 0: self.tile_process() # 瓦片处理或整体处理
        else: self.process()
        output_img_rgb_tensor = self.post_process() # 后处理

        # 3. 输出RGB转换 (Tensor CHW RGB -> NumPy HWC BGR)
        # ... (tensor转numpy, clamp, 通道和维度转换, 转回灰度如果原始是灰度) ...

        # 4. 处理Alpha通道 (如果存在)
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan': # 使用模型放大alpha
                # ... (对alpha通道执行 pre_process -> tile_process/process -> post_process) ...
                # ... (将输出的3通道alpha转回单通道灰度) ...
            else: # 使用线性插值放大alpha
                output_alpha = cv2.resize(alpha, (w_input * self.scale, h_input * self.scale), ...)
            # 合并RGB和Alpha
            output_img = cv2.cvtColor(output_img_bgr, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # 5. 反归一化和类型转换 (转回 uint8 或 uint16)
        # ... (乘以 max_range, round, astype) ...

        # 6. 最终缩放 (如果 outscale 与模型 scale 不同)
        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(output, (int(w_input * outscale), int(h_input * outscale)), ...)

        return output, img_mode
```

*   **输入处理**:
    *   将输入的NumPy图像（BGR顺序）转换为内部处理所需的RGB顺序，并归一化到 `[0, 1]` 范围。
    *   支持8位和16位图像输入。
    *   能处理灰度图像（临时转为RGB处理，结果转回灰度）和RGBA图像（分离alpha通道）。
*   **RGB通道处理**: 调用 `pre_process`, `tile_process`/`process`, `post_process` 对RGB数据进行超分。
*   **Alpha通道处理**:
    *   如果输入是RGBA图像，可以选择如何处理alpha通道：
        *   `alpha_upsampler='realesrgan'`: 将alpha通道也视为一个灰度图，用同样的RealESRGAN模型进行超分（需要先将其转为3通道RGB格式喂给模型，再将结果转回单通道）。
        *   其他（例如默认或`'cv2'`）: 使用OpenCV的 `cv2.resize` 进行简单的双线性插值放大。
    *   处理后，将放大的alpha通道合并回图像。
*   **输出处理**:
    *   将处理后的张量转回NumPy数组，颜色通道从RGB转回BGR（OpenCV常用格式）。
    *   根据原始图像位深，将像素值从 `[0, 1]` 反归一化到 `[0, 255]` (uint8) 或 `[0, 65535]` (uint16)。
    *   如果提供了 `outscale` 参数且与模型本身的 `self.scale` 不同，则使用 `cv2.resize` (Lanczos4插值) 将图像调整到最终期望的输出尺寸。
*   **返回**: 返回处理后的NumPy图像和原始图像的模式字符串。

## 3. `PrefetchReader` 和 `IOConsumer` 类

这两个类主要用于优化批量推理时的文件读写性能。

### 3.1. `PrefetchReader`

```python
class PrefetchReader(threading.Thread):
    def __init__(self, img_list, num_prefetch_queue):
        super().__init__()
        self.que = queue.Queue(num_prefetch_queue) # 内部队列
        self.img_list = img_list # 待读取的图像路径列表

    def run(self): # 线程启动时执行
        for img_path in self.img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # 读取图像
            self.que.put(img) # 放入队列
        self.que.put(None) # 结束标志

    def __next__(self): # 实现迭代器协议
        next_item = self.que.get()
        if next_item is None: raise StopIteration
        return next_item

    def __iter__(self): return self
```

*   **目的**: 在一个单独的后台线程中预先读取图像文件。当主线程需要下一张图像时，可以直接从队列中获取，而不需要等待磁盘IO，从而避免阻塞主线程的计算（如模型推理）。
*   **工作方式**:
    1.  初始化时接收一个图像路径列表和队列大小。
    2.  `run` 方法在后台线程中遍历路径列表，使用 `cv2.imread` 读取图像，并将读取到的图像对象放入内部的 `queue.Queue`。
    3.  当所有图像读取完毕，放入一个 `None` 作为结束信号。
    4.  通过实现 `__iter__` 和 `__next__` 方法，`PrefetchReader` 对象本身可以作为迭代器使用。

### 3.2. `IOConsumer`

```python
class IOConsumer(threading.Thread):
    def __init__(self, opt, que, qid): # opt可能包含输出目录等配置
        super().__init__()
        self._queue = que # 存储待保存结果的队列
        self.qid = qid # 线程ID (用于日志)
        self.opt = opt

    def run(self):
        while True:
            msg = self._queue.get() # 从队列获取消息
            if isinstance(msg, str) and msg == 'quit': break # 退出信号

            output = msg['output'] # 图像数据
            save_path = msg['save_path'] # 保存路径
            cv2.imwrite(save_path, output) # 保存图像
        print(f'IO消费者线程 {self.qid} 已完成。')
```

*   **目的**: 在一个单独的后台线程中将处理完成的图像保存到磁盘。这使得主线程在输出结果后可以不必等待磁盘写入完成，即可继续处理下一张图像。
*   **工作方式**:
    1.  初始化时接收一个配置对象 `opt`（可能未使用）、一个用于传递待保存数据的队列 `que`，以及一个线程ID `qid`。
    2.  `run` 方法在后台线程中循环，不断从队列中获取消息。
    3.  消息通常是一个字典，包含处理好的图像数据 (`output`) 和目标保存路径 (`save_path`)。
    4.  使用 `cv2.imwrite` 将图像保存到磁盘。
    5.  当从队列中获取到字符串 `'quit'` 时，线程结束。

## 4. 总结

`realesrgan.utils.py` 提供了 Real-ESRGAN 项目进行高效、灵活推理的关键组件。`RealESRGANer` 类是核心，它封装了从模型加载到图像输出的完整超分辨率流程，支持瓦片处理、半精度、DNI模型插值以及对不同图像模式（灰度、RGB、RGBA）的处理。`PrefetchReader` 和 `IOConsumer` 则是通过多线程优化批量处理中IO瓶颈的实用工具，分别用于图像的预读取和异步保存。这些工具共同确保了 Real-ESRGAN 模型能够被方便且高效地应用于实际的图像增强任务。
