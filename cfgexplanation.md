为确保我准确解释“训练参数”的含义，我将快速查看本项目中 RL 配置的数据类，提取可见的字段列表（算法、策略、runner 与训练控制），然后总结每个参数的作用与建议范围。预计产出：一份与仓库一致的参数说明。

读取 [](file:///home/mast/mymjlab/mjlab/src/mjlab/rl/config.py)

先给出路线图：我会先解释什么是 MJCF，然后分层说明本项目里与“训练”相关的配置参数含义与调参要点——按策略网络参数、PPO 算法参数、Runner/训练控制参数、训练脚本参数（设备/视频/恢复）、以及速度任务环境的关键参数来讲，尽量结合仓库里的实际字段。

## 什么是 MJCF？

- MJCF 是 MuJoCo 的 XML 模型描述格式（MuJoCo XML, 通常称作 MJCF），用结构化的 XML 来描述机器人和场景。
- 你会在 MJCF 里看到：
  - assets：网格、纹理等资源
  - worldbody/bodies：刚体、层级关系、质心/惯量
  - joints：转动/滑动关节，限位、阻尼、摩擦等
  - geoms：碰撞几何体、材质、摩擦系数、接触维度（condim）
  - actuators：驱动器（电机/力矩源）与关节映射
  - sensors：触地、接触力、加速度等传感器
  - tendons/equality/keyframe：肌腱、约束以及关键帧姿态等
- 在本项目里，通常通过 `mujoco.MjSpec.from_file("<robot>.xml")` 读取 MJCF，随后补齐 `assets` 并在运行时编译为 `MjModel` 使用。示例参考：g1_constants.py 的 `get_spec()`。

这意味着：要替换机器人，核心是准备好 MJCF 和 assets，并在相应的 `<your_robot>_constants.py` 里提供 `get_spec()`、执行器/碰撞/初始位姿等配置，项目就能把你的模型装入统一的 RL 任务框架。

## 强化学习训练的“各种参数”分层解读

下面基于仓库实际代码，按层次说明关键参数。路径参考：config.py、train.py、以及速度任务通用配置 velocity_env_cfg.py。

### 1) 策略网络参数 RslRlPpoActorCriticCfg

- init_noise_std：策略动作高斯噪声初始标准差。越大探索越强，但早期容易发散；常见 0.5–1.0。
- noise_std_type：噪声类型，"scalar" 或 "log"；一般保持默认。
- actor_obs_normalization / critic_obs_normalization：是否对观测做归一化（actor/critic 各自）。关掉可减少额外状态，但在观测尺度变化大时打开有助稳定。
- actor_hidden_dims / critic_hidden_dims：MLP 隐层维度。越大表达力越强但更吃算力、易过拟合。
- activation：激活函数，如 "elu"、"tanh"、"relu"。

调参提示：
- 先用中等规模（如 256-256 或 512-256-128），收敛后再考虑缩放。
- 关节维度多/观测复杂时，增大网络有帮助；动作延迟/复杂性高时，合理加深 critic 网络也有帮助。

### 2) PPO 算法参数 RslRlPpoAlgorithmCfg

- num_learning_epochs：每次更新迭代训练多少 epoch（对一批轨迹反复利用）。过大可能过拟合。
- num_mini_batches：把一批数据分几份做小批量 SGD。mini batch 大小 = num_envs × num_steps_per_env / num_mini_batches。
- learning_rate：学习率。过大易发散，过小收敛慢。自适应调度时可略大起步。
- schedule：学习率调度，"adaptive" 或 "fixed"。adaptive 常与 desired_kl 搭配控制步长。
- gamma：折扣因子，0.99 常见。越接近 1 越重视长期回报。
- lam：GAE 的 λ，常见 0.95。协调 bias-variance。
- entropy_coef：熵系数（鼓励探索）。过大导致策略过于随机，过小易陷局部。
- desired_kl：期望新旧策略 KL 距离；配合 adaptive 调度自动调学习率，控制更新幅度。
- max_grad_norm：梯度裁剪阈值，防止梯度爆炸。
- value_loss_coef：价值函数损失权重；critic 训练不稳时可调大。
- use_clipped_value_loss：是否对 value loss 做裁剪（提升稳定性）。
- clip_param：PPO 的策略裁剪阈值（如 0.2）。过小更新太保守，过大不稳。
- normalize_advantage_per_mini_batch：是否按 mini-batch 归一化优势；通常全局归一化就够，必要时可开启。

调参提示（经验）：
- 采样批量（num_envs × num_steps_per_env）越大，学习率可相应略增。
- 出现 KL 过大/收敛抖动：减小学习率、减小 clip_param 或调大 desired_kl（让调度更保守）。
- 策略塌陷/早熟：增大 entropy_coef、增大优势归一化力度或扩大批量。

### 3) Runner/训练控制参数 RslRlOnPolicyRunnerCfg 与 RslRlBaseRunnerCfg

- seed：随机种子，便于复现。
- num_steps_per_env：每个环境每轮采多少步；与 env 数量共同决定每次更新的数据量。
- max_iterations：训练轮数（每轮：采样+更新+日志+存档）。
- save_interval：多少轮存一次 checkpoint。
- experiment_name：实验名；同时决定日志目录 `logs/rsl_rl/<experiment_name>/...`。
- run_name：可选的额外后缀（会出现在时间戳目录名里）。
- logger："wandb"/"tensorboard"；本项目默认 "wandb"。
- wandb_project：W&B 项目名，默认 "mjlab"。
- resume：是否从已有 run 恢复。
- load_run：选择要恢复的 run 目录（支持正则；默认 ".*" = 选最新）。
- load_checkpoint：要恢复的模型文件名（支持正则，默认 "model_.*.pt"）。
- clip_actions：动作裁剪范围（如设为 1.0 则在 [-1,1] 裁剪）；None 表示不裁剪。

关键关系：
- 有效批量 = num_envs × num_steps_per_env。调 `num_steps_per_env` 或并行 env 数量都会改变学习稳定性/速度。
- save_interval 决定磁盘占用与回滚粒度。训练不稳时可以调小保存频率以便回溯。

### 4) 训练脚本参数 TrainConfig（train.py）

- env：环境配置（如 `LocomotionVelocityEnvCfg` 的具体子类）。
- agent：上面 runner 的配置（含 policy/algorithm）。
- registry_name：仅 tracking 任务需要，用于从 WandB 拉取 motion artifact。
- device："cuda:0"/"cpu" 等。
- video：是否在训练中开启视频录制（RecordVideo，离线保存，默认不上传到 W&B）。
- video_length：每段视频最多帧数（以环境步计）。
- video_interval：多少步触发开始录制一次（默认 2000）。若训练步数很少或间隔太大，可能一次都不触发。
- 其他行为：训练开启 video 时会把 env 的 render_mode 设为 "rgb_array"，不弹窗。想即时看窗口，建议用 `play.py --viewer native`。

实践建议：
- 想要训练过程中也产生视频：把 `--video` 打开，并降低 `--video-interval`（注意 I/O 影响）。
- 更推荐“训练后用 play 录一段视频”：对训练无干扰，也更可控。

### 5) 速度任务环境关键参数（`LocomotionVelocityEnvCfg`）

来自 velocity_env_cfg.py：
- scene.num_envs：并行环境数（通过命令行 `--env.scene.num-envs` 可改）。越大采样越快，但显存/内存消耗更高。
- decimation：控制频率（环境步到控制步的下采样比）；例如 4 表示 200 Hz 仿真、50 Hz 控制（取决于 timestep）。
- episode_length_s：每个 episode 的持续时间（秒）。
- sim.mujoco.{timestep, iterations, ls_iterations}：物理步长与求解迭代；影响稳定性与速度。
- commands.twist：速度指令范围/重采样时间等
  - ranges.lin_vel_x/lin_vel_y/ang_vel_z：线/角速度目标的采样范围
  - resampling_time_range：多久重采样一次指令
  - heading_command/heading_control_stiffness：给定朝向与收敛强度
- rewards.*：各个奖励项及权重（如速度跟踪、姿态正则、动作率惩罚、关节限位等）。训练目标改变时，主要通过这里调权重。
- events.*：重置、随机扰动、摩擦/材质随机化等
- curriculum.*：课程学习，逐步加难（如地形难度、速度范围）
- viewer：渲染视角配置
- actions.joint_pos.scale：动作缩放（关节命名正则到幅度映射，通常从机器人常量里计算出如 `G1_ACTION_SCALE`）

调参要点：
- 初期可先平地（flat_env_cfg），关闭部分课程与强随机扰动；收敛后再开 rough/curriculum。
- rewards 权重非常关键：先保证“速度跟踪”能学起来，再逐步加入姿态/耗能等正则与任务约束。
- commands 的范围决定任务难度；从窄到宽逐步扩展（可配 curriculum 里的 velocity_stages）。

## 典型参数搭配与经验

- 批量设定：num_envs × num_steps_per_env 约在 1e4~1e5 级别通常比较稳（视显存与模型大小而定）。
- 学习率：自适应（adaptive）+ desired_kl=0.01 是一个稳妥的默认；KL 经常过大就减 lr 或减小 clip_param。
- 熵系数：0.005–0.02 常见；探索不够就稍增大，出现“乱动”就减小。
- critic 不稳（价值爆炸/误差大）：增大 value_loss_coef 或减小学习率、加大批量与正则。
- 视频录制：尽量用 play 评估而不是训练时录，避免训练速度受 I/O 影响。

## 小结

- MJCF 是 MuJoCo 的 XML 机器人/场景描述格式；本项目用 `MjSpec.from_file` 载入你的 MJCF 并统一装入 RL 框架。
- 训练参数分四层：
  - 策略网络（网络结构、噪声、是否归一化）
  - PPO 算法（学习率/调度、GAE、熵、KL、裁剪、梯度裁剪）
  - Runner/训练控制（批量步长、迭代轮次、存档/恢复、日志）
  - 脚本与环境（设备/视频、并行环境数、物理仿真、指令范围、奖励/事件/课程）
- 建议从平地、较小范围的指令和稳健的默认超参开始，先对齐“能跑起来并稳定增长”，再逐步加难/调优。

如果你愿意，我可以：
- 根据你当前任务，给一份“保守稳定”的 PPO/RL 配置清单；
- 或根据你的 GPU/内存，推荐具体的 num_envs 与 num_steps_per_env 组合；
- 帮你把视频上传到 W&B（在 train.py/play.py 自动 wandb.log 视频）。


# 训练参数OBS

我先分别回答三个点：什么是“recurrent”（循环/带记忆）模型、什么是 ONNX 的 metadata、以及它为什么能给出输入/输出的“顺序”。

## Recurrent 是什么（循环/带记忆的策略）
- 含义：recurrent 策略指带“记忆状态”的网络（RNN/LSTM/GRU）。除了当前观测 obs 外，它还接收上一步的隐藏状态 h（LSTM 还要 c），输出本步动作和新的隐藏状态。
- 推理接口（本仓库导出器的约定）：
  - 非 recurrent：输入只有 obs，输出 actions。
  - LSTM：输入 ["obs", "h_in", "c_in"]，输出 ["actions", "h_out", "c_out"]。
  - GRU：输入 ["obs", "h_in"]，输出 ["actions", "h_out"]。
- 影响 sim2sim：如果你的策略是 recurrent，你的外部推理循环必须自己管理 h/c：
  1) 第一步把 h_in/c_in 初始化为 0；
  2) 每步推理后保存 h_out/c_out；
  3) 下一步作为 h_in/c_in 传回模型。

相关代码：exporter.py 中 `_OnnxPolicyExporter.export()` 明确了输入/输出名字和形状。

## Metadata 是什么（模型内的键值元信息）
- 含义：ONNX 文件允许附加自定义的键值对（metadata）。本仓库在导出后调用 `attach_onnx_metadata(...)` 把“环境里实际使用的顺序与映射”写进模型：
  - observation_names：policy 组的观测项名称顺序
  - command_names：命令项名称（例如 ["twist"]）
  - joint_names：关节/执行器顺序（动作如何对应到关节）
  - action_scale：动作缩放（raw_action * scale + offset 的 scale 部分）
  - 还包括 default_joint_pos、joint_stiffness/damping、run_path 等
- 读取方式（Python 简例）：
  ```python
  import onnx
  m = onnx.load("policy.onnx")
  meta = {p.key: p.value for p in m.metadata_props}
  print(meta["observation_names"])  # 逗号分隔字符串，按需 split(',')
  ```
- 相关代码：exporter.py 的 `attach_onnx_metadata()`。

## 它为什么能给出“顺序”（来源于训练环境当时的真实顺序）
- 观测顺序来自 ObservationManager：
  - `env.observation_manager.active_terms["policy"]` 就是训练中“policy 组”的观测项列表（按拼接顺序）。导出器直接把这个列表写入 metadata 的 `observation_names`。
- 动作顺序与缩放来自 ActionManager/JointAction：
  - 动作维度的切片和顺序由 `ActionManager` 中 term 的注册顺序 + `JointAction.find_actuators()` 的返回顺序决定；导出器把 `joint_names` 和 `action_scale` 写入 metadata，保证外部可以一一对齐。
- 命令名称同理：`env.command_manager.active_terms` 写入 `command_names`。

也就是说：导出 ONNX 时，导出器“快照”了训练环境内实际使用的观测/动作/命令的顺序与映射，你在部署端读回这些 metadata，就能无歧义地重建 obs 拼接和 action 对应表。这就是“为什么它能给出顺序”。

## 最小实践提示
- 非 recurrent：
  - 直接把拼好的 obs（按 observation_names 顺序）喂给 ONNX，取出 actions（形状 = 关节数）。
- recurrent（LSTM/GRU）：
  - 初始化 h_in（和 c_in）为 0；
  - 每步把 (obs, h_in[, c_in]) 喂给 ONNX，得到 (actions, h_out[, c_out])；
  - 下步把 h_out/c_out 回传。
- 对齐检查：
  - 用 metadata 的 observation_names 决定 obs 拼接顺序；
  - 用 joint_names 决定关节/控制量的顺序；
  - 用 action_scale 决定 raw_action 到目标位姿的缩放（若训练时用了 offset=default_joint_pos，也要应用）。

如果你需要，我可以：
- 检测你的 ONNX 是否是 recurrent，并给出一段维护 h/c 的 wrapper 代码；
- 读取你的 ONNX metadata，生成一份“观测切片与动作映射”的表格，方便你把外部仿真数据无缝对齐。