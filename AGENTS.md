<!-- AUTONOMY DIRECTIVE — DO NOT REMOVE -->
YOU ARE AN AUTONOMOUS CODING AGENT. EXECUTE TASKS TO COMPLETION WITHOUT ASKING FOR PERMISSION.
DO NOT STOP TO ASK "SHOULD I PROCEED?" — PROCEED. DO NOT WAIT FOR CONFIRMATION ON OBVIOUS NEXT STEPS.
IF BLOCKED, TRY AN ALTERNATIVE APPROACH. ONLY ASK WHEN TRULY AMBIGUOUS OR DESTRUCTIVE.
<!-- END AUTONOMY DIRECTIVE -->

# AGENTS.md

本分支只交付网球 `hit` / `bounce` 事件检测的生产推理程序。

## 固定边界

- 三个核心输入：原视频、TrackNetV5 逐帧 CSV、自包含 deployment checkpoint。
- 唯一公开模型类：`TrajectoryPatchModel`，架构固定为 trajectory + 五帧 patch + ResNet-18 layer4 微调。
- 只允许推理所必需的在线轨迹构造、时间轴读取、patch 截取、归一化、前向和后处理。
- 禁止训练、数据增强、标注、离线预处理、评估、交叉验证、可视化验证、阈值搜索和实验代码。
- 禁止磁盘图像特征、patch、embedding 或时间轴缓存；只允许进程退出即消失的有界内存 LRU。
- 禁止 YAML、模型工厂、动态插件、兼容多个模型类或从研发包 `tennis_event` 导入代码。
- 同架构换权重只替换 checkpoint；架构变化必须替换唯一模型并升级 `model_id`。

## 工程约束

- 不硬编码数据集名、视频名、绝对路径、FPS 或分辨率；从 CLI、CSV 和视频元信息读取。
- 不新增无必要依赖、抽象、配置层或兼容分支；优先标准库和现有依赖。
- 输入边界必须严格校验，模型权重必须 `strict=True` 加载，错误不得静默降级。
- 推理只发布 `events.json` 和 `events.csv`，写出过程必须原子化。
- 所有文档使用中文；源码最多 8 个 Python 文件、1,500 行非空行。
- 新行为先写失败测试，再写最小实现；提交前运行相关 pytest、Ruff 和 compileall。
- 权重、视频、轨迹、输出、缓存和临时文件不得提交 Git。

## Git 提交

每个提交遵循仓库 Lore Commit Protocol，至少说明约束、拒绝方案、置信度、范围风险和验证证据。
