# AME 算法移植到 mjlab（Unitree Go1）说明

> 移植来源：[`AME_locomotion`](https://github.com/SII-FUSC/AME_Locomotion)（论文 *Attention-Based Map Encoding for Learning Generalized Legged Locomotion* 的 G1 复现）
> 移植目标：mjlab `AME-1` 分支（v1.6.0），Unitree Go1（12-DoF 四足）velocity 任务
> 移植原则：**只改 mjlab，零改 rsl_rl 源码**；AME 模型通过 `class_name` 注入标准 PPO。

---

## 1. AME 算法简介

AME（Attention-Based Map Encoding）用一个**注意力地形编码器**替代传统的高度扫描向量，让策略显式地对局部地形做空间注意力。

数据流：

```
proprio (2D) ──────────────────► proprio_normalizer ──► query_proj ──► query
                                                                      │
terrain_points [B,H,W,3] ──► CNN(local_encoder) ──► token 序列 ──────► MultiheadAttention
                            (可选 concat 坐标)         │                  │
                                                       └─(可选 global)──► │
                                                                         ▼
                            latent = cat(proprio, attention_out, [global])
                                  ▼
                            MLP head ──► action distribution (actor) / value (critic)
```

- `terrain_points` 是机器人中心化的地形点云 `[B, num_x, num_y, 3]`，三通道为 yaw 系下命中点相对 sensor 原点的位移（x 前向、y 侧向、z 高度）。
- 两个变体：`paper`（z-only 输入、concat 坐标、无下采样、无全局上下文）；`g1`（xyz 输入、CNN 下采样、可选全局上下文）。**本次移植采用 g1 变体配置**（xyz + 下采样 + 关闭全局上下文）。

---

## 2. 移植决策（已与用户确认）

| 决策点 | 选择 | 理由 |
|---|---|---|
| Reward 体系 | Go1 原生四足 reward | AME 原始 reward 为 G1 双足专用（手臂/腰部/双足协调），四足不适用；AME 贡献在编码器而非 reward |
| 代码组织 | 新建 `tasks/velocity_ame/` 任务包 | 与 PPO 的 go1 任务隔离，参照 `velocity_amp`/`velocity_vision` 模式 |
| 变体/阶段 | 单 go1 变体 + base/finetune 两阶段 | 保留 AME 泛化训练流程（finetune 的域随机化 + 地图漂移是关键） |
| 对称性 | 首版关闭（`symmetry_cfg=None`），保留接口 | G1 对称性映射不适用四足，需重写；先跑通再加 |
| rsl_rl | 用 uv 管理的 5.4.2，不改源码 | 5.4.2 已含 AME 所需全部特性（symmetry/share_cnn/construct_algorithm） |

---

## 3. 向 mjlab 添加的文件

```
src/mjlab/tasks/velocity_ame/
├── __init__.py                  # 包说明
├── velocity_ame_env_cfg.py      # AME 基础工厂（四组观测 + map_drift，机器人无关）
├── mdp/
│   ├── __init__.py              # 导出 terrain_points、resample_map_scan_drift
│   ├── observations.py          # terrain_points 观测（[B,H,W,3] 点云）
│   └── events.py                # resample_map_scan_drift（finetune 地图漂移）
├── rl/
│   ├── __init__.py              # 导出 AmeOnPolicyRunner
│   ├── modules.py               # AmeBaseModel/AmeActorModel/AmeCriticModel + 导出 wrapper
│   ├── runner.py                # AmeOnPolicyRunner
│   └── attention_viewer.py      # AmeAttentionViewer（play 时注意力可视化）
├── scripts/
│   ├── __init__.py
│   ├── play.py                  # AME play 入口（--attention 开关）
│   └── viz_terrain.py           # 地形可视化（ame-viz-terrain）
└── config/
    ├── __init__.py
    ├── variants.py              # go1 base/finetune 任务规格
    └── go1/
        ├── __init__.py          # 注册 Mjlab-VelocityAme-Base/Finetune-Unitree-Go1
        ├── rl_cfg.py            # AmeModelCfg/AmePpoAlgorithmCfg/AmeRunnerCfg
        └── env_cfgs.py          # Go1 特化（robot/sensors/friction/reward）

docs/ame_port/PORTING.md         # 本文档
```

**未修改任何 mjlab 既有文件、未修改 rsl_rl 源码**（仅在 `docs/source/changelog.rst` 加一条 changelog）。

---

## 4. 各组件说明

### 4.1 模型 `rl/modules.py`

从 AME 的 `actor_critic.py` 原样移植。`AmeBaseModel` 是独立 `nn.Module`（不继承 rsl_rl 的 `MLPModel`），但构造签名 `(obs, obs_groups, obs_set, output_dim, ...)` 与 `PPO.construct_algorithm` 的调用约定一致，因此可被标准 PPO 直接实例化。

- 支持两种观测布局：**多组模式**（`proprio_obs_groups` + `terrain_obs_group`，Go1 使用）和嵌套模式（单组 TensorDict）。
- `share_cnn_encoders=True` 时，`PPO.construct_algorithm` 把 actor 的 `cnns` dict 传给 critic 的 `cnns` 参数，actor/critic 共享 CNN+attention 编码器（仅 query_proj 因 proprio 维度不同可能不共享）。
- 提供 `as_jit`/`as_onnx`/`export_attention_metadata`，支持 TorchScript/ONNX 导出与注意力可视化。

### 4.2 观测 `mdp/observations.py: terrain_points`

从 RayCastSensor 生成机器人中心化地形点云 `[B, num_x, num_y, 3]`：

1. 用 `GridPatternCfg.generate_rays` 缓存网格形状与标称偏移；
2. `hit_pos_w - pos_w` 得命中点相对位移，`yaw_quat` + `quat_apply_inverse` 旋到 yaw 系；
3. 未命中射线的 x/y 用标称偏移、z 用 `-max_distance` 哨兵；
4. 可选高度噪声、地图漂移、高度裁剪。

**必须配合 `ray_alignment="yaw"`**（Go1 rough 的 `terrain_scan` 已是此配置）。Go1 的 `terrain_scan` 是 `GridPatternCfg(size=(1.6, 1.0), resolution=0.1)` → 17×11 射线，与 AME paper 变体网格吻合，输出 `[B, 17, 11, 3]`。

### 4.3 事件 `mdp/events.py: resample_map_scan_drift`

每个 env 采样一个 XY 漂移向量（零均值高斯，std=0.02m），挂在 `env._ame_map_scan_drift_xy`，由 `terrain_points(apply_drift=True)` 读取并加到地形图 x/y 通道，模拟系统性定位偏差。仅 finetune 阶段启用。

### 4.4 Runner `rl/runner.py: AmeOnPolicyRunner`

继承 `VelocityOnPolicyRunner`（已含 ONNX 导出 + metadata）：

- **`save`**：在父类保存后，额外导出 actor/critic 的 `attention_metadata.json`（记录 proprio 维度、terrain 形状、注意力形状、编码器变体等）和 `ame_runner_metadata.json`（记录 runner/actor/critic 类名），供下游注意力可视化使用。
- **`load`**：跳过 `MjlabOnPolicyRunner` 的 legacy 键迁移（`actor.*→mlp.*`、`actor_obs_normalizer.*→obs_normalizer.*`）。AME 模型 state_dict 直接存 `proprio_normalizer`/`local_encoder`/`attention`/`mlp`/`distribution` 等子模块，无 `actor.` 前缀，不适用该迁移；仅保留 rsl-rl 4.x→5.x 的 `std→distribution.std_param` 迁移与 `common_step_counter` 恢复。

### 4.5 配置 `config/go1/rl_cfg.py`

- `AmeModelCfg(RslRlModelCfg)`：新增 `map_latent_dim`、`num_attention_heads`、`terrain_input_mode`、`concat_coords_post_cnn`、`cnn_downsample`、`attach_global`、`proprio_obs_groups`、`terrain_obs_group` 等字段。
- `AmePpoAlgorithmCfg(RslRlPpoAlgorithmCfg)`：新增 `symmetry_cfg: dict | None = None`（首版 None，接口保留）。
- `unitree_go1_ame_runner_cfg(phase)`：actor/critic 用 `class_name="mjlab.tasks.velocity_ame.rl.modules:AmeActorModel/AmeCriticModel"`，`share_cnn_encoders=True`，`obs_groups` 分 actor/critic 各含 proprio+terrain 两组；base 15000 iters、finetune 3200 iters。

### 4.6 环境（两层结构）

AME 环境配置分两层，与 mjlab velocity 任务对称（`velocity_env_cfg.py` 基础
工厂 + `config/<robot>/env_cfgs.py` 机器人特化）：

**基础工厂 `velocity_ame_env_cfg.py`**（机器人无关）：

`make_velocity_ame_env_cfg(phase, play)` **完整独立定义所有配置项**（与
`velocity_env_cfg.py` 的 `make_velocity_env_cfg()` 对称，不依赖它），包括：
scene/terrain、sensors（terrain_scan 17×11、foot_height_scan）、observations、
actions、commands（twist）、events、rewards（13 项四足 reward）、terminations、
curriculum（terrain_levels + command_vel）、metrics、viewer、sim、decimation。

其中 AME 特化部分：

- 观测替换为 AME 四组（`terrain_points` 点云替代 `height_scan`）：
  - `actor_proprio`：base_ang_vel、projected_gravity、joint_pos、joint_vel、actions、command（**不含 base_lin_vel**，actor 须从地形+proprio 推断运动）
  - `actor_terrain`：terrain_points（finetune 加高度噪声 + 漂移）
  - `critic_proprio`：上述 + base_lin_vel + foot_height/air_time/contact/contact_forces（特权信息）
  - `critic_terrain`：terrain_points（无噪声/漂移）
- finetune 阶段加 `map_drift`（机器人无关，挂在 env 属性）
- num_envs、episode_length 按 phase/play 设置

**Go1 特化 `config/go1/env_cfgs.py`**：

`unitree_go1_ame_env_cfg(phase, play)` 调 `make_velocity_ame_env_cfg()` 后添加
Go1 机器人特化（与 go1 rough 一致，沿用 Go1 原生四足 reward 体系）：

- sim 参数、robot asset、terrain_scan frame=trunk
- 5 个 contact sensors（feet/self/thigh/shank/trunk_head）
- foot_friction 三轴（condim 6）、base_com、action scale、viewer
- reward：pose std、body/site 名字、关闭 body_ang_vel/angular_momentum/air_time、加碰撞惩罚
- terminations：删 fell_over、加 illegal_contact（大腿触地）、terrain curriculum
- finetune 加 `trunk_mass`/`trunk_inertia`（trunk body，Go1 特化）
- play 模式覆盖（删 push_robot、curriculum={}、randomize_terrain、terrain 5×5）

### 4.7 任务注册 `config/go1/__init__.py`

mjlab 用 import-time 注册（`import mjlab.tasks` 时 `import_packages` 递归 import 子包触发 `register_mjlab_task`），无 entry-points。因此 `__init__.py` 在模块顶部直接遍历 `iter_task_specs()` 注册两个任务：

- `Mjlab-VelocityAme-Base-Unitree-Go1`
- `Mjlab-VelocityAme-Finetune-Unitree-Go1`

---

## 5. 重大调整（G1 → Go1）

| 项 | AME G1 原版 | Go1 移植版 |
|---|---|---|
| 机器人 | Unitree G1（29-DoF 双臂人形） | Unitree Go1（12-DoF 四足） |
| Reward | 双足专用（手臂/腰部/双足空中时间/肩-髋协调等 21 项） | Go1 原生四足（速度跟踪/姿态/足部清除/摆动/滑动/碰撞等） |
| 体命名 | torso_link、ankle_roll_link、pelvis | trunk、FR/FL/RR/RL |
| 关节 | hip/knee/ankle/shoulder/elbow/wrist/waist | hip/thigh/calf（每腿 3 关节） |
| 地形观测 | terrain_points [B,17,11,3]（paper）/[B,33,21,3]（g1） | terrain_points [B,17,11,3]（沿用 Go1 terrain_scan 17×11） |
| 地形配置 | 自定义 pallets/gaps/beams 等 family | Go1 原生 ROUGH_TERRAINS_CFG |
| 对称性 | augment_g1_symmetry（G1 关节镜像表） | 关闭（symmetry_cfg=None），待写四足镜像表后启用 |
| Runner launch helper | 自带 ame-train/play/eval CLI（tyro） | 移除，用 mjlab 原生 `python -m mjlab.scripts.train` |
| obs_normalization | False（proprio_normalizer 自处理） | 沿用 False（保持 AME 风格） |

---

## 6. AME 在 mjlab 中的实现机制

### 6.1 模型注入链路

```
RslRlOnPolicyRunnerCfg.actor.class_name = "mjlab.tasks.velocity_ame.rl.modules:AmeActorModel"
        │ (asdict 序列化)
        ▼
OnPolicyRunner.__init__ → PPO.construct_algorithm
        │ resolve_callable(class_name)  # 冒号限定名 → importlib 解析
        ▼
actor = AmeActorModel(obs, obs_groups, "actor", env.num_actions, **cfg["actor"])
        │ share_cnn_encoders=True → cfg["critic"]["cnns"] = actor.cnns
        ▼
critic = AmeCriticModel(obs, obs_groups, "critic", 1, cnns=actor.cnns, **cfg["critic"])
        │
        ▼
alg = PPO(actor, critic, storage, symmetry_cfg=None, ...)
```

`resolve_callable`（`rsl_rl/utils/utils.py`）支持 `module:Attr` 冒号限定名，把字符串解析为类。AME 模型实现了 PPO 期望的全部接口（`forward`/`get_latent`/`update_normalization`/`output_*`/`get_output_log_prob`/`get_kl_divergence`/`is_recurrent`），故无需继承 `MLPModel`。

### 6.2 观测分组

`obs_groups = {"actor": ("actor_proprio","actor_terrain"), "critic": ("critic_proprio","critic_terrain")}`。`resolve_obs_groups` 解析后，AME 模型 `_get_obs_dim` 检测到多组 + 4D terrain 张量，自动进入多组模式：把 proprio 组拼接、terrain 组作为 `[B,H,W,3]` 点云。

### 6.3 训练流程

1. **base**：`Mjlab-VelocityAme-Base-Unitree-Go1`，4096 envs，rough 地形课程，15000 iters，从零训练。
2. **finetune**：`Mjlab-VelocityAme-Finetune-Unitree-Go1`，在 base checkpoint 上续训，3200 iters，加 map_drift + trunk 质量/惯量随机化 + 地形扫描噪声/漂移，硬化地形编码器。

finetune 通过 `--resume` + `--load-run` 加载 base checkpoint。`AmeOnPolicyRunner.load` 恢复模型权重与 `common_step_counter`（保留课程状态）。

---

## 7. 环境与依赖

### 7.1 rsl_rl 版本

- `pyproject.toml`/`uv.lock` 声明 `rsl-rl-lib==5.4.2`（PyPI），已含 AME 所需全部特性（`symmetry_cfg`、`share_cnn_encoders`、`construct_algorithm`、`extensions/symmetry.py`）。
- **AME 移植零改 rsl_rl**，直接用 5.4.2 标准特性。

### 7.2 venv 污染清理（重要）

`src/rsl_rl/` 是 AMP_vision 分支遗留的 editable 5.0.1 副本（v1.2.0 适配）。venv 里的 `__editable__.rsl_rl_lib-5.0.1.pth` 会让 `import rsl_rl` 加载它而非 uv 声明的 5.4.2，且 `uv run`/`uv sync` **不会自动清理**（editable dist-info 让 uv 认为 rsl-rl-lib 已满足）。验证前需手动清理：

```sh
rm .venv/lib/python3.13/site-packages/__editable__.rsl_rl_lib-5.0.1.pth
rm .venv/lib/python3.13/site-packages/__editable___rsl_rl_lib_5_0_1_finder.py
rm -r .venv/lib/python3.13/site-packages/rsl_rl_lib-5.0.1.dist-info
uv sync
uv run python -c "import rsl_rl; print(rsl_rl.__file__)"  # 应指向 .venv/.../site-packages/rsl_rl
```

`src/rsl_rl/` 目录保留不动（属 AMP_vision，git 未跟踪）。代码对 5.0.1/5.4.2 的 AME 相关 API 均兼容，清理可在验证前做。

### 7.3 未来在 rsl_rl 上二次开发

若需改 rsl_rl，不要直接改 site-packages，也不要复用 `src/rsl_rl/`。用 uv sources 指向 AME 分支专属副本：

```toml
[tool.uv.sources]
rsl-rl-lib = { path = "vendor/rsl_rl", editable = true }  # 基于 5.4.2 fork
```

---

## 8. 训练命令

```sh
# base 训练（4096 envs，15000 iters）
uv run python -m mjlab.scripts.train Mjlab-VelocityAme-Base-Unitree-Go1 \
  --env.scene.num-envs 4096
```

## 8.1 断点续训（resume）

AME 支持断点续训：从已有 checkpoint 恢复模型权重、迭代计数
（`current_learning_iteration`）与环境步数计数（`common_step_counter`，保留
地形课程状态），继续增量训练。已实测 save -> load -> 续训闭环通过（见第 10 节）。

**机制**：`--agent.resume True` 触发加载；`--agent.load-run`（正则，默认 `.*`
取最新 run 目录）、`--agent.load-checkpoint`（正则，默认 `model_.*.pt` 取最新
checkpoint）定位文件。`AmeOnPolicyRunner.load` 恢复权重与计数器后，`learn()`
从恢复的迭代号增量训练。

**关键注意点**：

- `--agent.max-iterations` 是**增量**而非总目标。rsl-rl 的 `learn()` 从
  `current_learning_iteration` 起跑 `max_iterations` 次：从 `model_10000.pt`
  续训、`--agent.max-iterations 15000` 会训到 iter 25000 而非 15000。若想训到
  原定 15000，需 `--agent.max-iterations 5000`。实测：从 `model_1.pt` 续训
  `--agent.max-iterations 1`，日志显示 `Learning iteration 1/2`（start_it=1,
  total_it=2）。
- tyro 的 bool 参数必须显式传值：`--agent.resume True`。裸 `--agent.resume`
  会报 `invalid choice ... Expected one of ('True', 'False')`（即便后跟
  `--agent.load-run` 也一样报错）。
- resume 在 `<log_root>/<experiment_name>/` 下找 checkpoint，按任务的
  `experiment_name` 定位（base 为 `go1_velocity_ame_base`，finetune 为
  `go1_velocity_ame_finetune`）。`get_checkpoint_path` 在该目录下用 `load-run`
  正则匹配子目录、`load-checkpoint` 正则匹配文件名，均取字母序最新。

```sh
# 同任务断点续训（最常见）：从最新 run 的最新 checkpoint 续训 base
uv run python -m mjlab.scripts.train Mjlab-VelocityAme-Base-Unitree-Go1 \
  --env.scene.num-envs 4096 --agent.resume True

# 指定 run 与 checkpoint（load-run/load-checkpoint 均为正则）
uv run python -m mjlab.scripts.train Mjlab-VelocityAme-Base-Unitree-Go1 \
  --env.scene.num-envs 4096 --agent.resume True \
  --agent.load-run '2026-08-11_' --agent.load-checkpoint 'model_10000.pt'

# 续训到原定目标：从 model_10000.pt 再训 5000 次到 iter 15000
uv run python -m mjlab.scripts.train Mjlab-VelocityAme-Base-Unitree-Go1 \
  --env.scene.num-envs 4096 --agent.resume True --agent.max-iterations 5000
```

**finetune 从 base checkpoint 续训**：finetune 的 `experiment_name` 是
`go1_velocity_ame_finetune`，resume 默认在该目录下找 checkpoint，而 base
checkpoint 在 `go1_velocity_ame_base/` 下，跨目录找不到。需把 base 的 run
目录软链到 finetune 的 log_root 下，让 `load-run` 能匹配到：

```sh
# 1. 软链 base 的 run 目录到 finetune log_root（run 目录名按实际替换）
ln -s "$PWD/logs/rsl_rl/go1_velocity_ame_base/2026-08-11_12-00-00" \
  "$PWD/logs/rsl_rl/go1_velocity_ame_finetune/base_2026-08-11"

# 2. finetune resume：load-run 匹配软链目录名，取其中最新 checkpoint
uv run python -m mjlab.scripts.train Mjlab-VelocityAme-Finetune-Unitree-Go1 \
  --env.scene.num-envs 4096 --agent.resume True --agent.load-run 'base_'
```

## 8.2 可视化程序：AME Play 与注意力可视化

AME 专属 play 工具让你在 viewer 中回放训练好的策略，并可选地实时可视化
注意力编码器正在关注的地形位置。

### 文件结构

- `velocity_ame/scripts/play.py`：play 入口（tyro CLI），负责加载
  checkpoint、构造 env/policy、按 `--attention` 开关选 viewer。
- `velocity_ame/rl/attention_viewer.py`：`AmeAttentionViewer`，在 native
  viewer 上叠加注意力球体。

### 两种模式

| 模式 | 开关 | viewer | 说明 |
|---|---|---|---|
| 正常 play | `--no-attention` | `NativeMujocoViewer` | 标准回放，只显示机器人+地形 |
| 注意力可视化 | `--attention`（默认） | `AmeAttentionViewer` | 叠加彩色球体显示注意力分布 |

### 注意力可视化原理

AME actor 的 `forward`（`_latent_from_tensors`）每步缓存两个张量：

- `last_attention_weights` `[B, N]`：cross-attention 对 N 个地形 token 的权重（多头取平均）。
- `last_attention_points` `[B, H', W', 3]`：注意力 token 的坐标（机器人中心化 yaw 系，相对 sensor 原点）。

`AmeAttentionViewer` 继承 `NativeMujocoViewer`，重写 `_update_debug_visualizers`：

1. 先调 `super()` 跑 env 原有 debug 可视化（如 terrain_scan 射线）。
2. 从 `self.policy`（`get_inference_policy` 返回 AME actor 本身）取上述两个张量。
3. **坐标转换**：注意力点是机器人中心化的 yaw 系坐标，需用 terrain_scan sensor
   的世界位姿转回世界坐标才能画到 viewer：
   `world = sensor.pos_w + quat_apply(yaw_quat(sensor.quat_w), points)`
   （仅 yaw 旋转，z 不变，再加 sensor 世界位置）。
4. **颜色与大小均基于绝对权重**（非每帧 min-max 相对值），以均匀基线 `1/N`
   为锚点，固定映射范围，跨帧/跨 iteration 可比：
   - `scale = w_i × N`：相对均匀基线的倍数，`1.0` = 均匀（未学到聚焦），
     `>1` = 超基线被注意。
   - 颜色：`scale` 经固定范围 `[0, 6]` 线性映射蓝->红（`scale=1` 淡蓝、
     `scale≥6` 满红），表示绝对注意力强度。
   - 半径：`scale` 经 `sqrt` 阈值化映射，`scale≤1` -> 最小半径 0.01（低于
     均匀，视觉不在场），`scale≥6` -> 最大半径 0.08。显著性粗筛。
5. 用 `MujocoNativeDebugVisualizer.add_sphere` 画 N 个球体（DECOR 类别）。

这样训练初期（attention 均匀）全场淡蓝小球，训练后期少数红大球跳出，可一眼
判断 attention 是否从均匀演化到聚焦。min-max 相对值会丢失集中程度（均匀时也
红蓝分明），已弃用。

实测：17×11 地形网格经 CNN 下采样到 9×6 = 54 个 token，每帧画 54 个球。

### 使用命令

```sh
# 注意力可视化（默认，需本地图形显示）
uv run ame-play \
  Mjlab-VelocityAme-Base-Unitree-Go1 \
  --checkpoint-file /abs/path/to/model_15000.pt

# 正常 play（不画注意力，与 mjlab 原生 play 一致）
uv run ame-play \
  Mjlab-VelocityAme-Base-Unitree-Go1 --no-attention

# 从最新 run 自动找 checkpoint（不传 --checkpoint-file）
uv run ame-play \
  Mjlab-VelocityAme-Base-Unitree-Go1
```

常用参数：`--checkpoint-file`（指定 checkpoint 路径）、`--load-run`（正则匹配
运行目录）、`--attention`/`--no-attention`、`--viewer auto|native|viser`、
`--num-envs`、`--device`。

### 注意事项

- 注意力可视化只支持 **native viewer**（需图形显示）。服务器无 GUI 用 Xvfb，
  或 `--no-attention --viewer viser` 走 web viewer（viser 不支持注意力叠加）。
- 可视化看 env 0 的注意力分布（`--num-envs >1` 时仍只画 env 0）。
- 球体颜色：蓝=低/低于均匀基线，红=显著超基线；半径随绝对权重倍数增大
  （`scale≤1` 最小、`scale≥6` 最大）。颜色与大小均基于绝对权重，跨帧可比。
- 首步之前无注意力数据（actor 还没 forward），球体不显示。

---

## 8.3 地形配置

AME Go1 训练用 mjlab 原生 `ROUGH_TERRAINS_CFG`（Go1 rough 任务继承），
AME env_cfg 不修改地形，仅复用。

### 地形世界网格（TerrainGeneratorCfg）

- 每块地形 **8m × 8m**，外围 border 20m
- **课程模式**（`curriculum=True`）：7 种地形各占一列，**10 行难度递进**
  （`num_cols` 在课程模式下被强制为 `len(sub_terrains)=7`）
- 机器人按 episode 表现升降级（`terrain_levels_vel` 课程）

### 7 种地形类型与难度

| 地形 | 比例 | 关键参数 | 难度 |
|---|---|---|---|
| flat | 0.2 | 平地 | 易 |
| pyramid_stairs | 0.2 | 台阶高 0–0.1m，宽 0.3m | 易-中 |
| pyramid_stairs_inv | 0.2 | 同上（下行楼梯） | 易-中 |
| hf_pyramid_slope | 0.1 | 坡度 0–1.0（≈0–45°） | 中-难 |
| hf_pyramid_slope_inv | 0.1 | 同上（下行斜坡） | 中-难 |
| random_rough | 0.1 | 噪声 0.02–0.10m | 易 |
| wave_terrain | 0.1 | 振幅 0–0.2m，4 波 | 中 |

均为**连续地形**（楼梯/斜坡/粗糙/波浪），无离散障碍。

### 地形观测网格（terrain_scan）

- `RayCastSensorCfg` + `GridPatternCfg(size=(1.6, 1.0), resolution=0.1)`
  -> **17 × 11 射线**
- 覆盖机器人前方 1.6m × 侧向 1.0m
- `ray_alignment="yaw"`（射线随机器人 yaw 旋转、保持水平），`frame="trunk"`
- AME `terrain_points` 观测输出 `[B, 17, 11, 3]`（xyz 点云）喂给注意力编码器

### 课程

**`terrain_levels_vel`（地形难度升降级）**

实现：`mjlab/tasks/velocity/mdp/curriculums.py:terrain_levels_vel`。每个 episode
reset 时，按机器人本 episode 走的距离判定升降级：

- **走的距离**：`distance = norm(root_pos_w[env_ids,:2] - env_origins[env_ids,:2])`
  （相对出生点的水平位移）
- **升级（move_up）**：`distance > terrain_generator.size[0] / 2`
  （走的距离 > 地形块半长，即 8m 块的 4m——机器人成功穿越半块地形）
- **降级（move_down）**：`distance < 命令速度 × max_episode_length_s × 0.5`
  （走的距离 < 预期距离的一半——机器人没跟上指令）
  - `move_down *= ~move_up`（升级优先，不重复判降级）
- **初始 reset 冻结**：`if common_step_counter == 0: move_up = move_down = 0`
  （首次 reset 前机器人还在出生位，distance 无意义，冻结避免误升级）
- 调 `terrain.update_env_origins(env_ids, move_up, move_down)` 更新各行 level
- 返回 `mean`/`max` 全局 level + 每种地形类型（列）的平均 level（课程模式下
  列 = 地形类型，便于监控哪种地形练到几级）

**与之前版本（v1.2.0）的改进点**：

| 改进 | 版本 | 说明 |
|---|---|---|
| 课程模式每种地形一列 | v1.3.0 #811 | `num_cols` 强制 = `len(sub_terrains)`，每种地形占一列；`proportion` 从控制列数改为控制出生分布 |
| 行间难度确定性 | v1.5.0 #1027 | 课程模式地形难度在行间确定性生成，且能到达 `difficulty_range` 端点（之前不精确） |
| 初始 reset 冻结 | v1.5.3 #1094 | 修复首次 reset 把所有 env 从 level 0 提到 1（忽略 `max_init_terrain_level`）的 bug——即上面 `common_step_counter == 0` 的冻结逻辑 |
| per-terrain-type level 监控 | v1.6.0 | 返回每种地形类型的平均 level（课程模式列 = 类型），tensorboard 可看各类型进度 |

**地形布局与难度机制**

课程模式下地形是 `num_rows × num_cols` 网格（每块 `size` 米，AME 为 8m×8m）：

- **行 = 难度等级**：行 0 = `difficulty_range` 下界（最易），行 `num_rows-1` = 上界（最难）。
  行间难度线性插值：`difficulty = lower + (upper-lower) * row/(num_rows-1)`
  （`terrain_generator.py:256-257`）。AME 的 `difficulty_range=(0.0, 1.0)`、
  `num_rows=10` -> 10 级难度（0.0, 0.111, ..., 1.0）。
- **列 = 地形类型**：课程模式下列数 = 地形种类数（每种地形一列）。

**机器人走动与课程判定的关系**：

- 地形网格**物理连通**（8m×8m 块拼接，无墙），机器人 episode 中可以走到相邻地块。
- 但课程判定**只看 episode 结束时的行走距离**（相对出生点），不看机器人走到哪块。
- `terrain_types`（列）在训练中**固定**（按 `proportion` 分配后不变），升降级只改
  `terrain_levels`（行）。所以每个 env 始终在同一种地形类型上练，只变难度等级。
- reset 时回到 `env_origin`（当前 level/type 块的出生点），升级到更高难度的行或
  降级到更低难度的行（同列）。

**`difficulty_range` 端点**

`difficulty_range = (lower, upper)`（默认 `(0.0, 1.0)`，`TerrainGeneratorCfg` 字段），
定义地形难度的最小/最大值。课程模式下映射到行：行 0 生成 `difficulty=lower` 的地形
（最易），行 `num_rows-1` 生成 `difficulty=upper` 的地形（最难）。"端点"即 `(lower, upper)`
这两个极值。

v1.5.0 改进：之前版本行间难度不精确（可能到不了端点），现在确定性线性插值，能准确
到达 `difficulty_range` 的两个端点。

**`command_vel`（速度指令 3 阶段递进）**

- step 0: lin_vel_x ±1.0, ang_vel_z ±0.5
- step 120000: lin_vel_x -1.5~2.0, ang_vel_z ±0.7
- step 240000: lin_vel_x -2.0~3.0

### AME 是否需要更难地形？

AME 注意力编码器的优势在**需要精确足部放置的离散障碍地形**（跳跃、踩石、
窄梁）。当前 `ROUGH_TERRAINS_CFG` 全是连续地形，**没有离散障碍**，难以充分
发挥 AME 注意力优势。AME 原版 finetune 就用 pallets/gaps/beams/stones 等
离散地形。

mjlab 已有更难地形的 preset 但未被 `ROUGH_TERRAINS_CFG` 使用（在
`mjlab/terrains/config.py`）：

| preset | 关键参数 | 说明 |
|---|---|---|
| `stepping_stones` | 石块 0.4–0.8m，间距 0.2–0.5m，高 0.2m | 踩石，需精确落足 |
| `narrow_beams` | 梁 0.2–0.8m 宽，高 0.2m | 窄梁行走 |
| `discrete_obstacles` | 40 个障碍，高 0.05–0.3m | 避障 |
| `box_random_grid` | 网格高 0–0.3m | 网格跳跃 |
| `random_spread_boxes` | 80 个箱，高 0.05–0.3m | 散布障碍 |
| `open_stairs` / `random_stairs` | 台阶高 0.1–0.3m | 进阶楼梯 |
| `perlin_noise` | 高度 0–1m | 复杂噪声 |

**建议**：若要让 AME 注意力编码器发挥优势，可在
`velocity_ame/config/go1/env_cfgs.py` 里把 `terrain_generator` 替换为含离散
障碍的自定义配置（追加 `stepping_stones`/`narrow_beams`/`discrete_obstacles`
等）。但当前连续地形已能训练（smoke test 通过），是否加更难地形取决于训练目标。

### 地形可视化（ame-viz-terrain）

训练地形与 play 地形不同（train 用课程模式 10×7，play 用随机 5×5），所以 play
时看到的地形与训练时不一致。`ame-viz-terrain` 工具可单独可视化地形 mesh
（不训练、不创建机器人），用于查看训练/play 地形的真实样子：

```sh
# 看训练地形（课程 10×7，与训练时完全一致）
uv run ame-viz-terrain --mode train

# 看 play 地形（随机 5×5）
uv run ame-viz-terrain --mode play
```

实测地形规模：

| 模式 | 布局 | geom 数 |
|---|---|---|
| train | curriculum=True，10 行 × 7 列 = 70 块 | 554 |
| play | curriculum=False，5 行 × 5 列 = 25 块 | 197 |

实现：`velocity_ame/scripts/viz_terrain.py`，用 AME env_cfg 的 terrain 配置
（`ROUGH_TERRAINS_CFG` + `curriculum=True` for train / 5×5 for play）构造
`TerrainEntity` 生成 mesh，`mujoco.viewer.launch(terrain.spec.compile())` 显示。
需本地图形显示。

---

## 9. 风险与后续

| 风险/待办 | 说明 |
|---|---|
| checkpoint load 键迁移 | `AmeOnPolicyRunner.load` 已跳过不适用的 MLPModel 迁移；smoke test 验证 save（`model_*.pt` 生成），断点续训已实测通过（save -> load -> `learn()` 增量续训，`current_learning_iteration` 与 `common_step_counter` 正确恢复，日志 `Learning iteration 1/2` 确认从恢复迭代号起增量），见 8.1 节 |
| ONNX base metadata | `AmeOnPolicyRunner.save` 跳过 `get_base_metadata`（它访问 AME 不存在的 `"actor"` obs group，会抛 `KeyError`）；ONNX 文件与 attention metadata 正常生成，仅不附 base metadata |
| 对称性未启用 | `symmetry_cfg=None`；后续需为 Go1 写四足镜像表（FR↔FL、RR↔RL 关节镜像 + base yaw 翻转 + 重力 y 取反），填 `symmetry_cfg` 即可启用 |
| finetune reward 未调严 | AME G1 finetune 调严了多个 reward 权重；本次保持 Go1 原生，可按需调 `action_rate_l2`/`foot_clearance` 等权重 |
| 显存 | AME 模型含 CNN+attention，4096 envs 显存占用高于纯 MLP；按 GPU 调整 `--num-envs` |
| obs_normalization=False | 沿用 AME G1 风格；若训练不稳可尝试 `obs_normalization=True` |

---

## 10. 验证状态

- ✅ ruff format + lint 通过（`velocity_ame`，全项目无回归）
- ✅ ty 类型检查通过
- ✅ 任务注册成功（`Mjlab-VelocityAme-Base/Finetune-Unitree-Go1`）
- ✅ smoke test：64 envs × 2 iterations 训练通过（AME 模型构建、`terrain_points` `[B,17,11,3]` 观测、PPO 训练、checkpoint + ONNX + attention metadata 保存均正常，无警告）
- ✅ 断点续训实测通过（uv 管理的 rsl-rl 5.4.2，editable 5.0.1 污染已清理）：8 envs 跑 2 iter 生成 `model_*.pt`，`--agent.resume True` 续训 1 iter，成功加载 checkpoint 并从恢复的迭代号增量训练（`Learning iteration 1/2`），见 8.1 节
- ⚠️ 待验证：完整 15000-iter base 训练 + finetune 跨 experiment_name 软链 resume 仍未端到端跑通（断点续训机制已实测，长训练稳定性待验证）
