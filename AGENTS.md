# 奖励函数生成或修改
奖励函数要严谨，准确，有效，精简。奖励函数生成之后对你生成的代码进行详细复查。最后给予我逐行的代码讲解。
生成的奖励函数不要与现在存在在reward.py文件中的奖励函数有逻辑上的冲突。

# 本项目使用UV进行版本管理
相关依赖版本可以通过pyproject.toml进行查看

# 添加代码或修改代码
修改完成后要进行运行测试，进行最小冒烟测试
不要使用全局性的补丁代码，保证生成的代码有详细注释，保证可读性，规范性，风格统一。
检查自己的逻辑漏洞，进行自我审查，确保代码真实有效，所生成的带代码都被使用
最后阐明修改之前的细节是什么，原问题是什么。改动之后改进了什么。
说明改动后可能会存在什么漏洞，带来什么风险问题。

# 解释代码
我是一个初学者，对理论和代码的实现极度渴望。我希望当我需要你帮我详细说明代码的时候，先帮我理清楚：1.代码文件的目的，以及牵扯到算法的作用和意义。2.代码主要的算法原理，包括数学原理和算法原理。3.代码的循环实现结构，首先给出一个框图和伪代码，着重介绍代码是如何进行循环的。4.数据是什么时候被使用的，最重要的梯度和参数是如何被更新的，要结合算法的原理详细解释。5.结合前面的架构和框图，逐行解释每一行代码在整个循环的宏观作用，以及具体细节的微观作用。
# 调试与实证纪律（强制，来自 2026-10-15 site 渲染事故）

## 1. 先实证，后结论——禁止凭记忆断言
用户报告"看不到 X / Y 不生效"这类问题时，第一动作是**写最小验证脚本拿到
事实**（离屏渲染对比、打印运行时配置值、读绑定源码），而不是凭对库默认值
的记忆下结论。本次事故：用户问 "site 是否被渲染隐藏了？"，我凭"mjvOption
的 site 默认开启"的记忆回答"没有被隐藏"，实际上 **`sitegroup` 默认只有
group 0-1 参与渲染**，head site(group=5) 一直被跳过。用户两次反证（size
改大仍不可见）后我才做实证——一次 30 秒的离屏渲染像素对比即可定位。

## 2. 修复范围最小化——禁止"顺手全开"
定位到根因后，修复必须**只覆盖报告的问题**。本次事故：只需让一个 head
site 可见，我却把 `sitegroup[2..5]` 全部打开，导致 flat_patch 标记、
terrain origin 标记等几十个调试 site 涌进渲染画面，破坏整体可读性。
正确做法（事后验证有效）：改目标对象自身的 group 属性（XML 里
`group="5"` → `group="0"`），一行、零副作用。

## 3. 多方案时选副作用最小的，并在回复中说明副作用对比
修改共享渲染/配置层（viewer、mjvOption、全局预设）前，先列出该修改影响
的其他使用方，向用户说明后再动手。

## 4. 几何/朝向类修改必须数值验证渲染结果
修改 site/camera 的 pos/quat 后，用离屏渲染或正向运动学脚本验证实际
forward 向量与位置（本项目已有 `src/mjlab/scripts/check_camera.py`）。
本次相机 quat 事故：把 `(w,0,y,0)` 误当绕 x 轴旋转，实际是绕 y 轴——
两轮修改方向全错，直到写 check_camera.py 数值验证才纠正。

## 5. 类内编辑禁止跨出类体；冒烟必须覆盖真实构造链
在类体内插入/替换代码时，new_string 不得引入模块级新类——否则后续成员
会被静默切进新类（2026-10-16 distribution 事故：`_ClampedDeterministicOutput`
插在 GaussianDistribution 中间，input_dim/mean/std 等全部属性被挂到辅助类，
play 构造 runner 时 NotImplementedError）。模块级辅助类一律追加到文件尾部。
配套教训：stub-only 冒烟不等于验证——必须跑通真实构造链（rsl_rl runner
构建 + checkpoint 加载 + viewer 打开）才算数；play 报错当时 stub 全绿。

# 训练推送规则（强制）

## 1. 未经明确指示不得提交训练
推送集群训练（sbatch）**必须**由用户当次明确指示。排队动辄数小时，
擅自提交会直接浪费一轮排队。改动完成后的默认动作是"报告 + 等待指示"，
不是"顺手推一发"。

## 2. 单次训练最长 7 小时（除非重大架构更新）
在**没有重大架构更新**（网络结构、观测/动作空间、地形生成、环境数/批量
规模等影响训练的改动）的情况下，服务器推送训练的**最长训练时间统一为
7 小时**：

- sbatch 墙钟上限：`--time=7:00:00`（不再是 8h/12h）。
- `max_iterations` 要**按 7h 预算折算**，不要出现"设定 6000 轮但 7h 只能
  跑 4500 轮被砍"的情况；宁可少设轮数，让它自然跑完收尾。
- 保留周期性 checkpoint（`save_interval`），确保被墙钟砍掉时仍有可用模型。
- 若有重大架构更新、确实需要更长训练：**先说明理由并取得用户同意**，再
  突破 7h。

# 服务器训练推送操作手册（本目录，按此执行）

适用范围：在本仓库向集群（202510192923@202.38.252.203，repo
~/scut-2026-mymjlab/mymjlab，分区 gpuA800）推送任何训练时，按本手册逐步执行。

## 0. 前置事实
- 集群：sbatch 分区 gpuA800（1TB 内存节点，4×A800/节点）。
- 仓库同步方式：本地 commit -> `git push origin` + `git push server` ->
  服务器 `git merge --ff-only origin/feat/parkour_vision_AMP`（工作区在
  feat/parkour_vision_AMP 分支；push server 即可达，merge 兜底确认）。
- 训练入口：`src/mjlab/scripts/train.py`（tyro CLI，布尔参数必须
  `True/False` 首字母大写，小写会 53 秒快速失败）。
- Python 输出必须加 `-u`（否则 slurm 日志全缓冲，中途不可读）。

## 1. 推送前检查清单（逐项过）
1. `git status` 确认用户改动已全部识别并**原样保留**（不"顺手修"用户参数）。
2. `ruff check` 改动文件 + 相关 pytest（地形改动必跑
   `tests/test_parkour_terrains.py`）。
3. 配置冒烟：`load_env_cfg` + `load_rl_cfg` 读回关键参数逐一断言
   （weight/lerp/gamma 等）。
4. 本地是 CPU：任何 GPU-only 路径（tensor.numpy() 等）都测不出来——
   新课程/奖励函数必须自查 device 转换（2026-11-01 full_v1 教训）。
5. 磁盘配额：单 run checkpoint ~2.1GB（81.5MB × save_interval 频率），
   logs/rsl_rl 目录曾涨到 25GB 触顶导致训练写 ckpt 即崩。堆积后清理
   废弃 run（每个只留 final）。
6. **RESUME 续训语义**：`runner.load` 恢复迭代计数器 =>
   `max_iterations = 底座已训轮数 + 本次新增轮数`（设错只多训几百轮）。
   底座 checkpoint 必须含判别器状态（AMP 任务），纯 PPO ckpt 不兼容。
7. 同步后服务器 `git log --oneline -1` 确认 HEAD 一致。

## 2. sbatch 模板要点
- 模板历史：`.cfgaudit/parkour_amp_v*.sbatch`（最新版为准复制 sed）。
- 固定头：`--partition=gpuA800 --gres=gpu:1 --cpus-per-task=16 --mem=200G
  --exclude=g02n03,g04n05`（200G 账面占用偏大、实际 MaxRSS ~0.5GB，
  可下调到 32G 提高调度通过率——2026-10-31 分析，未落地）。
- `--time` 按轮数折算：纯 PPO ~3.2s/轮；AMP ~4.3s/轮；AMP+5列地形
  ~5.3s/轮。留 30-60 分钟收尾余量，宁可少设轮数。
- train.py 命令行：任务 id + `--env.scene.num_envs 4096
  --agent.logger tensorboard --agent.max_iterations N --agent.run-name X`。
- 日志 `--output` 指向仓库根，命名 `<exp>_<jobid>.log`。

## 3. 提交后验证
- `squeue` 确认 PD/R；开跑后**先看日志头部有没有 preflight/参数报错**
  （tyro 大小写、AMP 43D preflight 都在头几秒炸）。
- stdout 缓冲：没加 `-u` 时日志中途不可读，用 TB event 或 checkpoint
  mtime 判断进度。
- 磁盘写崩特征：`RuntimeError: ... inline_container` + `Errno 122`。

## 4. 训练中/后标准动作
- 读取进度：服务器 `.venv/bin/python` + EventAccumulator 解析 event 文件
  （标出名与 grep 日志不同：`Train/mean_reward`、`Train/mean_episode_length`、
  `Policy/mean_std`、`Loss/value`、`Curriculum/terrain_levels`、
  `Episode_Termination/*`、`Episode_Reward/*`）。
- 拉取 ckpt：scp 到本地镜像路径
  `src/mjlab/scripts/logs/rsl_rl/<experiment>/<run_dir>/`，
  **md5sum 双端核对**；torch.load 冒烟验证 keys/iter。
- TB 本地：`UV_CACHE_DIR=$PWD/.uvcache uv run --with "setuptools<81"
  tensorboard --logdir <abs path> --port 6007 --load_fast=false`
  （venv 自带 TB 被 setuptools 82 破坏；6006 常被残留实例占用）。
  本地目录**必须有 event 文件** TB 才显示该 run（只放 ckpt 不显示）。
- play：`uv run python src/mjlab/scripts/play.py <task_id>
  --checkpoint-file <本地路径>`。
