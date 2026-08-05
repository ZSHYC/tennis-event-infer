# 网球击球与落点事件纯推理工具

本仓库用于从网球比赛视频中定位两类事件帧：

- `hit`：球员击球；
- `bounce`：球落地。

当前唯一模型是 `TrajectoryPatchModel`：它联合使用 TrackNetV5 轨迹窗口与五张在线截取的球点
patch，图像骨干为 ResNet-18，训练时微调 layer4。仓库只包含生产推理链路，不包含训练、标注、
数据增强、离线预处理、评估、交叉验证、可视化、MDD、全帧分支或磁盘图像特征缓存。

## 运行前需要准备什么

一次推理需要三个核心输入：

1. 一段原始比赛视频；
2. 与该视频逐帧对应的 TrackNetV5 CSV；
3. 与当前代码兼容的 deployment checkpoint。

运行环境需要：

- Ubuntu、Linux 或 WSL；
- Python 3.10 或更高版本；
- Git；
- 当前私有 GitHub 仓库的读取权限；
- 下载模型时需要已登录的 GitHub CLI（`gh`）；
- GPU 可选。没有可用 CUDA 时可以使用 CPU，但通常更慢。

## 输入合同

输入合同不是建议，而是程序会严格检查的边界。任一输入不满足合同，推理都会以非零退出码失败，
不会猜测、补齐或静默降级。

### 1. 原始视频

`--video` 指向待检测的完整视频。视频必须满足：

- 文件存在，并且 OpenCV 能够打开和逐帧解码；
- 至少包含一帧；
- 宽度、高度和 FPS 都是有效正数；
- 整段视频的分辨率保持不变。

视频是以下信息的最终权威：

- 总帧数；
- 原图宽度和高度；
- FPS；
- 每一帧的时间戳。

程序优先使用视频解码器报告的逐帧 PTS 作为时间轴。只有当 PTS 不可用或不是严格递增时，才使用
`frame_number / FPS` 作为时间戳回退。因此不要用 CSV 中的 FPS 或自行估算的时间戳替代视频时间轴。

生产推理面对的是没有 GT 的新视频，所以输入必须覆盖完整视频；不能只截取某段 GT 范围。

### 2. TrackNetV5 CSV

`--trajectory` 指向 UTF-8 CSV。文件必须包含以下列名：

| 列名 | 类型 | 必须满足的规则 |
| --- | --- | --- |
| `frame_number` | 整数 | 从 `0` 开始，逐行严格覆盖到 `N-1`，不得重复、跳帧或乱序 |
| `detected` | 整数 | 只能是 `0` 或 `1` |
| `x_orig` | 浮点数或空 | `detected=1` 时必须是原视频像素坐标；`detected=0` 时可留空 |
| `y_orig` | 浮点数或空 | `detected=1` 时必须是原视频像素坐标；`detected=0` 时可留空 |
| `width` | 正整数 | 原视频宽度，逐帧一致，并且与视频一致 |
| `height` | 正整数 | 原视频高度，逐帧一致，并且与视频一致 |

`N` 必须等于视频实际可解码帧数。即使 TrackNetV5 在某一帧没有检测到球，该帧也必须保留一行，
并写成 `detected=0`；不能删除漏检帧，否则 CSV 时间轴会与视频错位。

最小合法示例：

```csv
frame_number,detected,x_orig,y_orig,width,height
0,1,960.5,412.25,1920,1080
1,0,,,1920,1080
2,1,978.0,425.5,1920,1080
```

坐标系定义：

- 原点 `(0, 0)` 位于原视频左上角；
- `x_orig` 向右增大，必须满足 `0 <= x_orig < width`；
- `y_orig` 向下增大，必须满足 `0 <= y_orig < height`；
- 坐标必须是有限数值，不能是 `NaN` 或无穷大；
- 坐标对应原视频分辨率，不能直接填写 TrackNet 输入缩放图上的坐标。

CSV 可以包含 TrackNetV5 产生的其他列，程序允许这些列存在但不会读取或送入模型。特别是
TrackNet 原始置信度默认不进入模型输入。

典型非法情况：

```text
从 frame_number=1 开始              -> 非法，必须从 0 开始
0, 1, 3, 4                          -> 非法，缺少第 2 帧
同一个 frame_number 出现两次        -> 非法
detected=2                          -> 非法，只允许 0/1
detected=1 但 x_orig/y_orig 为空     -> 非法
x_orig 等于 width                   -> 非法，已超出画面右边界
CSV 写 1280×720，视频为 1920×1080   -> 非法，分辨率不一致
CSV 有 1000 行，视频能解码 1001 帧   -> 非法，帧数不一致
```

模型为了在线截取 patch，可以在 checkpoint 允许的短时间范围内为漏检帧使用邻近位置或插值位置。
这是模型内部的 patch 定位行为，不会伪造对外事件坐标：输出中的 `x/y` 只取事件帧真实观测；事件帧
`detected=0` 时输出 `null`。

### 3. Deployment checkpoint

`--checkpoint` 指向自包含部署权重。当前正式文件是：

```text
trajectory_patch_epoch10_threshold_010.pt
```

它包含：

- `TrajectoryPatchModel` 权重；
- 模型结构参数；
- 轨迹特征 schema 和归一化统计；
- 轨迹窗口参数；
- 五帧 patch 的尺寸、填充值、时间偏移和位置有效范围；
- hit/bounce 阈值；
- NMS 半径和后处理合同。

因此运行时不需要 YAML，也不允许通过 CLI 临时覆盖阈值或 NMS。当前 checkpoint 的 `model_id` 必须是
`trajectory_patch_resnet18_layer4_v1`；权重使用 `strict=True` 加载，结构不匹配会立即失败。

## 安装

### 1. 克隆私有仓库

```bash
git clone https://github.com/ZSHYC/tennis-event-infer.git
cd tennis-event-infer
```

### 2. 安装运行依赖

```bash
python -m pip install .
```
pip 会读取 [pyproject.toml](pyproject.toml)，安装运行依赖，并注册
  `tennis-event-infer` 命令。

运行依赖只有 NumPy、OpenCV、PyTorch 和 torchvision。PyTorch 是否能够使用 CUDA，取决于当前环境
安装的 PyTorch 构建和 NVIDIA 驱动，不由本项目静默切换。

### 3. 可选：安装开发验证依赖

```bash
python -m pip install '.[dev]'
```

它与上一条命令的区别是：

- `.`：仍然安装当前项目；
- `[dev]`：同时安装 `pyproject.toml` 中的可选开发依赖，目前是 pytest 和 Ruff；
- 外层单引号：防止 Bash/Zsh 把方括号当作通配符解释。

只运行推理时执行 `python -m pip install .` 即可；准备修改代码或运行测试时再安装 `'.[dev]'`。

## 下载并校验模型

模型不进入 Git 历史，而是作为私有 GitHub Release `v1.0.0` 的附件提供。

先确认 GitHub CLI 已登录且账号有仓库读取权限：

```bash
gh auth status
```

下载模型：

```bash
mkdir -p models
gh release download v1.0.0 \
  --repo ZSHYC/tennis-event-infer \
  --pattern trajectory_patch_epoch10_threshold_010.pt \
  --dir models
```
- 或者可以手动从github网页release处下载权重

下载完成后必须校验 SHA-256：

```bash
echo "95a21b89f8991d955c2de5cc7fc8ed5ec2ac9698dd0a11965e421cde5367dce5  models/trajectory_patch_epoch10_threshold_010.pt" \
  | sha256sum -c -
```

成功时必须看到：

```text
models/trajectory_patch_epoch10_threshold_010.pt: OK
```

如果显示 `FAILED`、找不到文件或哈希不一致，不要继续推理；应重新下载并确认版本和文件名。

## 执行推理

```bash
tennis-event-infer \
  --video /path/to/video.mp4 \
  --trajectory /path/to/tracknetv5.csv \
  --checkpoint models/trajectory_patch_epoch10_threshold_010.pt \
  --output-dir outputs/video_name \
  --device auto \
  --batch-size 128
```

参数说明：

| 参数 | 是否必需 | 含义 |
| --- | --- | --- |
| `--video` | 是 | 原始完整比赛视频路径 |
| `--trajectory` | 是 | 与视频逐帧对应的 TrackNetV5 CSV 路径 |
| `--checkpoint` | 是 | deployment checkpoint 路径 |
| `--output-dir` | 是 | 本次推理结果目录 |
| `--device` | 否 | `auto`、`cpu` 或 `cuda`，默认 `auto` |
| `--batch-size` | 否 | 每次模型前向处理的中心帧数，必须是正整数，默认 `128` |

设备选择：

- `--device auto`：PyTorch 检测到可用 CUDA 时使用 GPU，否则使用 CPU；
- `--device cuda`：强制使用 CUDA；CUDA 不可用时直接报错，不会偷偷改用 CPU；
- `--device cpu`：强制使用 CPU。

`batch-size` 只影响运行资源和速度，不改变 checkpoint 内的模型、阈值或 NMS。更大的 batch 通常需要
更多显存；显存不足时应降低它，例如改为 `64`、`32` 或 `16`。

命令成功后会在终端打印：

```text
device: cuda
frames: 36325
events: 283
elapsed_seconds: 3814.52
events_json: /absolute/path/to/output/events.json
events_csv: /absolute/path/to/output/events.csv
```

- `device`：实际使用的设备；
- `frames`：处理的总帧数；
- `events`：阈值和 NMS 后保留的事件数；
- `elapsed_seconds`：总耗时，单位为秒；
- `events_json/events_csv`：两个正式输出文件的绝对路径。

上面的数字是命令输出格式示例；实际数值由输入视频、硬件和模型结果决定。

## 输出合同

`--output-dir` 最终只发布两个文件：

```text
events.json
events.csv
```

两者表达完全相同的事件，只是序列化格式不同。字段固定为：

| 字段 | JSON 类型 | CSV 表示 | 含义 |
| --- | --- | --- | --- |
| `frame_number` | 整数 | 整数 | 事件所在的零基帧号，第一帧为 `0` |
| `timestamp_seconds` | 浮点数 | 浮点数 | 事件帧在视频时间轴上的时间，单位为秒 |
| `event_type` | 字符串 | 字符串 | 只可能是 `hit` 或 `bounce` |
| `confidence` | 浮点数 | 浮点数 | `[0,1]` 内的模型事件分数；不是经过统计校准的真实概率 |
| `x` | 浮点数或 `null` | 数字或空单元格 | 事件帧真实观测的原视频像素横坐标 |
| `y` | 浮点数或 `null` | 数字或空单元格 | 事件帧真实观测的原视频像素纵坐标 |

JSON 示例：

```json
[
  {
    "frame_number": 2672,
    "timestamp_seconds": 56.00333333333334,
    "event_type": "hit",
    "confidence": 0.9400460720062256,
    "x": 896.25,
    "y": 303.75
  },
  {
    "frame_number": 2910,
    "timestamp_seconds": 60.74166666666667,
    "event_type": "bounce",
    "confidence": 0.8125,
    "x": null,
    "y": null
  }
]
```

对应 CSV：

```csv
frame_number,timestamp_seconds,event_type,confidence,x,y
2672,56.00333333333334,hit,0.9400460720062256,896.25,303.75
2910,60.74166666666667,bounce,0.8125,,
```

第二个事件的空坐标表示 TrackNetV5 在事件帧没有真实观测到球；它不代表坐标为 `(0, 0)`，也不代表
模型无法检测事件。

如果没有任何事件超过 checkpoint 阈值：

- `events.json` 内容为 `[]`；
- `events.csv` 仍会存在，但只有表头。

后处理与顺序：

- 每帧分别产生 hit 和 bounce 模型分数；
- hit 与 bounce 使用 checkpoint 内各自的阈值和独立 NMS；
- 最终事件按 `frame_number`、再按 `event_type` 排序；
- 同一帧理论上可以同时保留不同类型事件；
- 输出不包含逐帧原始分数、patch、embedding、缓存、评估指标或可视化。

写出保护：

- 输出先在同级临时目录完整写好，再原子替换正式目录；
- 写出失败时会恢复已有正式结果，避免只更新 JSON 或只更新 CSV；
- 如果已有输出目录包含 `events.json/events.csv` 以外的文件，程序拒绝覆盖，防止误删其他数据。

## 常见错误与排查

| 错误或现象 | 常见原因 | 处理方式 |
| --- | --- | --- |
| `Repository not found` | 没有私有仓库权限，或 GitHub 账号未登录 | 确认已被邀请，并检查 `gh auth status` |
| `release not found` | 版本名或仓库名错误 | 使用 `v1.0.0` 和 `ZSHYC/tennis-event-infer` |
| SHA-256 显示 `FAILED` | 下载不完整、文件损坏或拿错模型 | 删除该文件后重新下载，校验通过再运行 |
| `V5 CSV 缺少字段` | CSV 表头缺少六个必需列 | 按输入合同补齐准确列名 |
| `V5 CSV 必须逐帧覆盖 0..N-1` | 删除了漏检帧、帧号跳跃/重复/乱序 | 为每个视频帧保留一行，漏检写 `detected=0` |
| `检测帧坐标超出画面范围` | 使用缩放图坐标或宽高填写错误 | 转换回原视频像素坐标并核对分辨率 |
| `视频帧数与 V5 轨迹帧数不一致` | CSV 没覆盖完整视频，或视频版本不同 | 对同一视频重新导出完整逐帧 CSV |
| `视频与轨迹分辨率不一致` | CSV 坐标系不是原视频分辨率 | 修正 `width/height` 和 `x_orig/y_orig` |
| `device=cuda，但当前环境没有可用 CUDA` | PyTorch/驱动不支持 CUDA | 修复 CUDA 环境，或显式使用 `--device cpu` |
| `batch_size 必须是正整数` | batch size 为 0、负数或非整数 | 使用 `128` 或其他正整数 |
| `输出目录包含无关文件，拒绝覆盖` | 输出目录还保存了其他文件 | 换一个新输出目录，不要让程序覆盖混合目录 |

查看完整命令帮助：

```bash
tennis-event-infer --help
```

## 开发验证

先安装开发依赖：

```bash
python -m pip install '.[dev]'
```

然后运行：

```bash
pytest -q
ruff check src tests
python -m compileall -q src
python -m pip check
tennis-event-infer --help
```

- `pytest -q`：运行生产输入、checkpoint 和短视频端到端测试；`-q` 表示精简输出；
- `ruff check src tests`：检查源码和测试中的语法、未定义名称及选定规范问题；
- `python -m compileall -q src`：把源码编译为 Python 字节码，用于发现语法错误；
- `python -m pip check`：检查已安装依赖是否存在版本冲突或缺失；
- `tennis-event-infer --help`：确认安装后的命令入口和参数合同可用。

## 当前版本与实测信息

- 项目版本：`1.0.0`；
- GitHub Release：`v1.0.0`；
- 模型：epoch 10 `TrajectoryPatchModel` deployment checkpoint；
- 模型 ID：`trajectory_patch_resnet18_layer4_v1`；
- checkpoint SHA-256：
  `95a21b89f8991d955c2de5cc7fc8ed5ec2ac9698dd0a11965e421cde5367dce5`；
- 当前 checkpoint 的 hit/bounce 阈值均为 `0.10`，NMS 半径为 5 帧；
- 已在 NVIDIA GeForce RTX 5070 Ti Laptop GPU 上完成 36,325 帧完整视频推理，batch size 128，
  耗时约 63.6 分钟。

上述耗时只证明当前链路完成过全视频验证，不是其他机器的性能保证。实际速度取决于视频编码与
分辨率、磁盘解码速度、CPU、GPU、PyTorch/CUDA 环境和 batch size。
