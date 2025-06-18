# `inference_realesrgan_video.py` 代码分析

## 1. 文件概述

`inference_realesrgan_video.py` 脚本是 Real-ESRGAN 项目提供的用于视频超分辨率处理的命令行工具。它能够读取多种格式的视频文件（或图像序列文件夹），逐帧使用 Real-ESRGAN 模型进行放大和增强，并最终将处理后的帧合成为一个新的视频文件。该脚本依赖 `ffmpeg-python` 库进行视频的解码和编码，并支持通过多进程并行处理来加速较长视频的修复过程。

## 2. 主要组件与逻辑

### 2.1. 依赖与初始化

```python
import argparse, cv2, glob, mimetypes, numpy as np, os, shutil, subprocess, torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from os import path as osp
from tqdm import tqdm

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

try:
    import ffmpeg # 尝试导入 ffmpeg-python
except ImportError:
    import pip
    pip.main(['install', '--user', 'ffmpeg-python']) # 如果失败，则尝试安装
    import ffmpeg
```

*   **导入**: 导入了众多标准库以及 `basicsr` 和 `realesrgan` 的相关模块。
*   **FFmpeg**: 脚本的核心依赖之一是 `ffmpeg-python` 库，用于与 `ffmpeg` 程序交互以进行视频解码和编码。如果未安装，脚本会尝试通过 `pip` 进行安装。

### 2.2. 辅助函数

*   **`get_video_meta_info(video_path)`**:
    *   使用 `ffmpeg.probe(video_path)` 获取指定视频文件的元数据。
    *   提取并返回视频的宽度、高度、平均帧率（FPS）、是否有音频流（并创建音频流对象）以及总帧数。
    *   是后续处理（如视频分割、Reader/Writer初始化）的基础。

*   **`get_sub_video(args, num_process, process_idx)`**:
    *   用于多进程处理时，将原始输入视频分割成多个子片段。
    *   如果 `num_process` 为1，则直接返回原始视频路径。
    *   否则，根据总进程数 `num_process` 和当前进程索引 `process_idx`，计算该进程应处理的视频时间段。
    *   调用 `args.ffmpeg_bin`（ffmpeg命令行程序）使用 `-ss` (起始时间) 和 `-to` (结束时间) 参数从原始视频中提取对应的子片段，并保存为临时MP4文件。
    *   返回该子片段的路径。

### 2.3. `Reader` 类

此类负责从不同来源（视频文件、图像文件夹）读取帧数据。

```python
class Reader:
    def __init__(self, args, total_workers=1, worker_idx=0):
        # ... (参数保存) ...
        # 判断输入类型 (视频/图像文件夹/单图像)
        input_type_guess = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type_guess is None and osp.isdir(args.input) else input_type_guess

        if self.input_type is not None and self.input_type.startswith('video'):
            video_path = get_sub_video(args, total_workers, worker_idx) # 获取 (子)视频路径
            # 启动ffmpeg子进程，通过管道输出原始BGR24视频帧
            self.stream_reader = ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24', ...).run_async(...)
            meta = get_video_meta_info(video_path) # 获取元信息
            # ... (保存宽度, 高度, fps, 音频, 总帧数) ...
        else: # 图像文件夹或单张图像
            # ... (根据worker_idx分配图像路径列表 self.paths) ...
            # ... (从第一张图像获取宽度、高度) ...
        self.idx = 0 # 图像列表索引

    def get_frame_from_stream(self): # 从ffmpeg管道读取一帧
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)
        # ... (处理字节并reshape为numpy数组) ...

    def get_frame_from_list(self): # 从文件列表读取一帧
        # ... (cv2.imread) ...

    def get_frame(self): # 根据输入类型调用对应方法
    # ...
    def close(self): # 关闭ffmpeg流
    # ...
    # ... (其他辅助方法如 get_resolution, get_fps, get_audio, __len__) ...
```

*   **初始化**:
    *   根据输入路径的MIME类型或是否为目录来判断输入是视频、图像文件夹还是单张图像。
    *   **视频输入**: 调用 `get_sub_video` 获取当前工作进程应该处理的（可能是分割后的）视频路径。然后，启动一个 `ffmpeg` 子进程，该进程解码此视频并通过管道 (`pipe:`) 将原始视频帧（BGR24格式）输出到 `self.stream_reader.stdout`。同时，获取该视频的元数据。
    *   **图像输入**: 如果是图像文件夹，则将文件列表根据工作进程总数 (`total_workers`) 和当前进程索引 (`worker_idx`) 进行划分，使得每个进程处理一部分图像。如果是单张图像，则路径列表只包含该图像。从第一张图像获取宽度和高度。
*   **核心方法**:
    *   `get_frame()`: 根据输入类型，调用 `get_frame_from_stream()`（从ffmpeg管道读取）或 `get_frame_from_list()`（从文件列表用OpenCV读取）来获取下一帧。
    *   `close()`: 如果是视频流，则关闭 `ffmpeg` 子进程。
*   其他方法用于获取视频/图像序列的分辨率、FPS、音频流和总帧数。

### 2.4. `Writer` 类

此类负责将处理后的帧写入输出视频文件。

```python
class Writer:
    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale) # 计算输出帧尺寸
        # ... (输出超大分辨率警告) ...

        # 启动ffmpeg子进程，从管道接收原始BGR24视频帧，编码并写入输出文件
        if audio is not None: # 如果有音频流 (来自Reader)
            self.stream_writer = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}', framerate=fps)\
                                     .output(audio, video_save_path, pix_fmt='yuv420p', vcodec='libx264', acodec='copy', ...).run_async(...)
        else: # 无音频
            self.stream_writer = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}', framerate=fps)\
                                     .output(video_save_path, pix_fmt='yuv420p', vcodec='libx264', ...).run_async(...)

    def write_frame(self, frame): # 将帧写入ffmpeg管道
        self.stream_writer.stdin.write(frame.astype(np.uint8).tobytes())

    def close(self): # 关闭ffmpeg流
        self.stream_writer.stdin.close()
        self.stream_writer.wait()
```

*   **初始化**:
    *   根据原始帧尺寸和用户指定的 `--outscale` 计算输出视频帧的宽度和高度。
    *   启动一个 `ffmpeg` 子进程，该进程从管道 (`pipe:`) 接收原始视频帧（BGR24格式），并使用指定的编码器（如 `libx264`）、像素格式（如 `yuv420p`）和FPS将其编码到 `video_save_path` 指定的输出视频文件中。
    *   如果 `Reader` 提供了音频流 (`audio`)，则该音频流会直接复制 (`acodec='copy'`) 到输出视频中，保持音画同步。
*   **核心方法**:
    *   `write_frame(frame)`: 将处理好的NumPy图像帧转换为字节流，并写入到 `ffmpeg` 子进程的 `stdin`。
    *   `close()`: 关闭 `ffmpeg` 子进程的 `stdin`（表示没有更多帧了），并等待其完成编码和文件写入。

### 2.5. `inference_video` 函数 (核心处理单元)

这是执行单个视频（或视频片段）超分辨率处理的工作函数。在多进程模式下，每个进程会执行这个函数。

```python
def inference_video(args, video_save_path, device=None, total_workers=1, worker_idx=0):
    # 1. 模型选择与加载 (与 inference_realesrgan.py 类似)
    # ... (根据 args.model_name 实例化 RRDBNet 或 SRVGGNetCompact, 确定 netscale, file_url) ...
    # ... (确定 model_path, 如果不存在则从 file_url 下载) ...
    # ... (处理 realesr-general-x4v3 的 DNI 权重) ...

    # 2. 初始化 RealESRGANer
    upsampler = RealESRGANer(scale=netscale, model_path=model_path, dni_weight=dni_weight, model=model,
                             tile=args.tile, tile_pad=args.tile_pad, pre_pad=args.pre_pad,
                             half=not args.fp32, device=device) # device 参数由多进程调用时传入

    # 3. 初始化 GFPGAN (可选)
    # ... (如果 args.face_enhance, 则初始化 face_enhancer) ...
    # ... (对动漫模型禁用 face_enhance) ...

    # 4. 初始化 Reader 和 Writer
    reader = Reader(args, total_workers, worker_idx) # worker_idx 和 total_workers 用于视频分割
    audio = reader.get_audio() if worker_idx == 0 else None # 只有主worker处理音频
    height, width = reader.get_resolution()
    fps = reader.get_fps()
    writer = Writer(args, audio, height, width, video_save_path, fps)

    # 5. 逐帧处理循环
    pbar = tqdm(total=len(reader), unit='frame', desc=f'推理 worker {worker_idx}')
    while True:
        img = reader.get_frame()
        if img is None: break # 视频结束

        try:
            if args.face_enhance and face_enhancer:
                _, _, output = face_enhancer.enhance(img, ...)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error: # OOM等错误
            # ... (打印错误信息) ...
            continue # 跳过此帧
        else:
            writer.write_frame(output) # 写入处理后的帧

        if device: torch.cuda.synchronize(device) # GPU同步
        pbar.update(1)

    # 6. 清理
    reader.close()
    writer.close()
    pbar.close()
```

*   **模型加载与初始化**: 与 `inference_realesrgan.py` 中的逻辑基本一致，包括根据 `args.model_name` 选择网络架构、下载预训练权重、设置DNI（如果适用）。
*   **`RealESRGANer` 实例化**: 创建 `upsampler` 对象，用于实际的图像超分。
*   **GFPGAN 初始化**: 如果启用了面部增强，则初始化 `GFPGANer`。
*   **Reader/Writer**: 创建 `Reader` 和 `Writer` 实例来处理特定视频（或视频片段）的帧读写。在多进程模式下，`worker_idx` 和 `total_workers` 参数确保 `Reader` 正确读取分配给它的视频部分或图像序列。音频流通常只由第一个worker（`worker_idx == 0`）处理和传递给 `Writer`，以避免冲突。
*   **处理循环**:
    *   使用 `tqdm` 显示进度。
    *   循环调用 `reader.get_frame()` 获取帧。
    *   对获取到的帧调用 `upsampler.enhance()` (或 `face_enhancer.enhance()`) 进行处理。
    *   将处理后的帧通过 `writer.write_frame()` 写入输出视频流。
    *   包含错误处理逻辑。
*   **关闭**: 处理完成后关闭 `Reader` 和 `Writer`（即关闭ffmpeg子进程）。

### 2.6. `run` 函数 (多进程编排与单进程执行)

此函数是视频处理的主要协调者。

```python
def run(args):
    args.video_name = osp.splitext(os.path.basename(args.input))[0] # 获取视频基本名
    video_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4') # 最终输出路径

    # 可选：先将整个视频提取为PNG帧序列，然后按图像文件夹模式处理
    if args.extract_frame_first:
        # ... (使用ffmpeg命令行将视频转为 %08d.png 格式的图像序列) ...
        args.input = tmp_frames_folder # 更新输入为帧文件夹

    num_gpus = torch.cuda.device_count() # 获取可用GPU数量
    num_process = num_gpus * args.num_process_per_gpu # 计算总工作进程数

    if num_process == 1: # 单进程模式
        device = torch.device(0) if num_gpus > 0 else torch.device('cpu') # 选择设备
        inference_video(args, video_save_path, device=device)
        return

    # 多进程模式
    ctx = torch.multiprocessing.get_context('spawn') # 使用 'spawn' 方法创建进程，对CUDA更友好
    pool = ctx.Pool(num_process) # 创建进程池
    # ... (创建临时子视频输出目录) ...

    pbar = tqdm(total=num_process, unit='sub_video', desc='多进程推理')
    for i in range(num_process): # 为每个进程分配任务
        sub_video_save_path = osp.join(tmp_out_video_dir, f'{i:03d}.mp4')
        # 异步提交任务到进程池
        pool.apply_async(
            inference_video, # 工作函数
            args=(args, sub_video_save_path, torch.device(i % num_gpus), num_process, i), # 参数
            callback=lambda arg: pbar.update(1)) # 每完成一个子任务更新进度条
    pool.close()
    pool.join() # 等待所有进程完成
    pbar.close()

    # 合并所有处理完的子视频片段
    # ... (创建 vidlist.txt 文件，列出所有子视频路径) ...
    # ... (使用 ffmpeg -f concat -i vidlist.txt -c copy ... 命令合并视频) ...

    # 清理临时文件和文件夹
    # ... (删除 _out_tmp_videos, _inp_tmp_videos, _vidlist.txt) ...
```

*   **帧提取模式**: 如果 `args.extract_frame_first` 为真，脚本会首先使用 `ffmpeg` 将整个输入视频提取成一个包含所有帧的图像文件夹。然后，`args.input` 被修改为指向这个文件夹，后续的处理就变成了图像序列处理模式（`Reader` 会按图像文件夹方式读取）。
*   **进程数计算**: 根据可用的GPU数量和用户指定的 `num_process_per_gpu`（每个GPU上运行的进程数）来确定总的并行进程数 `num_process`。
*   **单进程执行**: 如果 `num_process` 为1，直接调用 `inference_video` 处理整个视频。
*   **多进程执行**:
    1.  使用 `torch.multiprocessing`（以 `spawn` 上下文）创建一个进程池。
    2.  为每个进程（worker）调用 `pool.apply_async` 来异步执行 `inference_video` 函数。每个 `inference_video` 调用会处理视频的一个片段（通过 `get_sub_video` 在 `Reader` 内部实现视频分割）或图像序列的一部分。`torch.device(i % num_gpus)` 用于轮流为进程分配GPU。
    3.  所有子视频片段处理完成后，`run` 函数会创建一个 `vidlist.txt` 文件，其中包含所有已处理子视频片段的路径。
    4.  最后，调用 `ffmpeg` 命令行工具，使用 `concat` demuxer 将这些子视频片段无损合并（`-c copy`）成最终的输出视频 `video_save_path`。
    5.  清理所有临时创建的文件夹和文件。

### 2.7. `main` 函数 (命令行入口)

```python
def main():
    parser = argparse.ArgumentParser()
    # ... (定义所有命令行参数，包括视频特有的如 --fps, --ffmpeg_bin, --extract_frame_first, --num_process_per_gpu) ...
    args = parser.parse_args()

    # ... (输入路径清理，输出目录创建) ...
    # ... (判断输入是否为视频) ...
    # ... (如果是flv视频，先用ffmpeg转为mp4) ...
    # ... (如果extract_frame_first但输入非视频，则禁用该选项) ...

    run(args) # 调用run函数开始处理

    # ... (如果extract_frame_first，清理临时帧文件夹) ...

if __name__ == '__main__':
    main()
```

*   定义了脚本的命令行参数，包括一些视频处理特有的选项，如：
    *   `--fps`: 指定输出视频的帧率。
    *   `--ffmpeg_bin`: `ffmpeg` 可执行文件的路径。
    *   `--extract_frame_first`: 是否先将视频提取为图像帧再处理。
    *   `--num_process_per_gpu`: 每个GPU上启动的并行处理进程数。
*   对输入路径进行一些预处理。
*   判断输入是否为视频文件。
*   对FLV格式的视频，先用系统 `ffmpeg` 命令将其转换为MP4，因为 `ffmpeg-python` 对FLV的直接处理可能存在问题或效率不高。
*   调用 `run(args)` 函数启动核心的视频处理流程。
*   如果使用了 `--extract_frame_first`，在处理完毕后清理临时提取的帧文件夹。

## 5. 总结

`inference_realesrgan_video.py` 是一个强大且灵活的视频超分辨率工具。它通过 `Reader` 和 `Writer` 类巧妙地利用 `ffmpeg-python` 和 `ffmpeg` 命令行实现了高效的视频帧读取和写入。核心的图像放大依然依赖 `RealESRGANer` 类。为了处理长视频和利用多GPU资源，脚本实现了复杂的多进程处理机制：将视频分割成片段，由不同进程在不同（或相同，轮流）GPU上并行处理这些片段，最后再将处理结果无缝合并。同时，它也提供了先将视频完全解成图像帧再处理的选项，这在某些情况下可能更稳定或方便。集成的GFPGAN功能也同样适用于视频中的人脸。这些特性使得该脚本能够应对多种视频增强需求。
