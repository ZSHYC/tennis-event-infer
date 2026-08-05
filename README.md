# 网球事件纯推理工具

本分支只包含 `hit` 与 `bounce` 的生产推理链路。模型使用轨迹窗口与五张原始球点 patch，图像骨干为
ResNet-18，仅对应 layer4 微调权重；不包含训练、评估、数据加工、可视化或磁盘特征缓存代码。

## 输入合同

命令只需要三个核心输入：

- 原始比赛视频。
- TrackNetV5 逐帧 CSV：必须逐帧覆盖 `0..N-1`，必需列为 `frame_number`、`detected`、
  `x_orig`、`y_orig`、`width`、`height`；允许存在其他列，但不会用于模型输入。
- 自包含 deployment checkpoint：包含模型权重、轨迹归一化、patch 合同、阈值、窗口和 NMS 参数。

视频是帧数、分辨率、FPS 和时间轴的最终权威；CSV 与视频不一致时直接报错。`detected=1` 时坐标
必须是原视频分辨率下的有限画面内坐标；`detected=0` 时坐标不会被读取。

## 安装

```bash
git clone https://github.com/ZSHYC/tennis-event-infer.git
cd tennis-event-infer
python -m pip install .
```

开发验证额外安装：

```bash
python -m pip install '.[dev]'
```

## 获取模型

模型不进入 Git 历史。具有私有仓库读取权限并已登录 GitHub CLI 后执行：

```bash
mkdir -p models
gh release download v1.0.0 \
  --repo ZSHYC/tennis-event-infer \
  --pattern trajectory_patch_epoch10_threshold_010.pt \
  --dir models
```

校验下载文件：

```bash
echo "95a21b89f8991d955c2de5cc7fc8ed5ec2ac9698dd0a11965e421cde5367dce5  models/trajectory_patch_epoch10_threshold_010.pt" \
  | sha256sum -c -
```

## 命令合同

```bash
tennis-event-infer \
  --video /path/to/video.mp4 \
  --trajectory /path/to/tracknetv5.csv \
  --checkpoint models/trajectory_patch_epoch10_threshold_010.pt \
  --output-dir /path/to/output \
  --device auto \
  --batch-size 128
```

`--device` 和 `--batch-size` 只控制运行资源，不改变模型语义。

- `--device auto`：有 CUDA 时使用 GPU，否则使用 CPU。
- `--device cuda`：明确要求 GPU；CUDA 不可用时直接报错。
- `--device cpu`：强制 CPU，结果语义相同但通常更慢。

本分支已用上面的命令和 batch size 128 完成 36,325 帧视频的 GPU 全量验证，耗时约 63.6 分钟。
实际耗时取决于分辨率、视频解码、GPU 和 batch size。

## 输出合同

输出目录只发布 `events.json` 与 `events.csv`。每个事件包含：

```text
frame_number,timestamp_seconds,event_type,confidence,x,y
```

`x/y` 是事件帧 TrackNetV5 的原图观测坐标；该帧未检出时为 `null`，不会用插值坐标冒充观测值。

输出目录采用原子替换：成功时只包含这两个正式文件；失败时不会发布一半的新结果，也不会写入
磁盘图像特征缓存。

## Checkpoint 与错误处理

当前部署文件的 `model_id` 必须是 `trajectory_patch_resnet18_layer4_v1`。阈值和 NMS 参数来自
checkpoint，CLI 不允许临时覆盖，以免同一权重产生不可追溯的不同结果。

后续可以替换同一架构的 checkpoint，无需修改生产代码；应在研发分支用
`scripts/export_deployment_checkpoint.py` 从可信训练产物重新导出。若模型架构变化，则必须同步替换
唯一的 `TrajectoryPatchModel`、升级 `model_id` 并重新做完整迁移验证，不引入模型工厂或兼容分支。

缺文件、CSV 字段/坐标错误、视频与 CSV 帧数或分辨率不一致、checkpoint 合同错误以及显式请求但
不可用的 CUDA 都会以非零退出码失败，不会静默降级。
