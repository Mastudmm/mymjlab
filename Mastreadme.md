## 顺序逻辑
从task文件夹开始看，底层可以先不看

velocity_env_cfg.py ---   SceneCfg  ---  {num_envs,env_spacing,terrain(```TerrainImporterCfg```),entities,snesors,extent}    
                    ---   ObservationCfg   --- PolicyCfg --- **builtin_sensor** 




### builtin sensor
只要 MJCF/MjSpec 里已经有名为 "robot/imu_lin_vel"、"robot/imu_ang_vel"、"feet_ground_contact"、"robot/root_angmom"、"nonfoot_ground_touch" 等传感器，你就可以在 builtin_sensor 里用同样的字符串。
· 在编写MJCF文件的时候，就要搞清楚mujoco支持哪种类型的传感器

用 builtin_sensor：当数据来自 MuJoCo 自带的传感器（IMU、接触、力、速度、帧姿态等），名字就是场景里注册的传感器名（通常带实体前缀，如 robot/...）。
用 joint_pos_rel：当你要的是“实体关节的相对位置/速度”这类由实体状态派生的数据，不依赖任何传感器名，而是通过 SceneEntityCfg("robot", joint_names=...) 指定选择与顺序。
### 基本命令
- MUJOCO_GL=egl
这是一个环境变量，在命令前设置，作用于后续启动的进程。
作用：指定 MuJoCo 的渲染后端为 egl（OpenGL 的一种无头渲染方式）


- mujoco原生渲染界面，命令的线速度为深蓝，实际线速度为青色，实际角速度为浅绿色
- 

### 降低显存的训练参数

- MUJOCO_GL=egl
  - 含义：用 EGL 作为 MuJoCo 的无头 GPU 渲染后端，适合服务器/无显示器环境。与显存压缩无直接关系，但让渲染可在没有窗口的环境中运行。
  - 出处：MuJoCo 环境变量（上游约定）。

- --env.scene.num-envs 1
  - 默认：各任务示例常用 4096（README 示例）/ 你本次设为 1
  - 影响：
    - 显存/内存：大幅降低（最直接的降显存手段之一）
    - 采样吞吐：极大降低，WALL 时间会变长，曲线更抖
    - 训练稳定性：小批量采样方差更大；可用更长训练时间或稍增 epochs/熵系数来对冲
  - 出处：`LocomotionVelocityEnvCfg.scene.num_envs`（velocity_env_cfg.py），可命令行覆盖

- --env.sim.nconmax 15000（默认 140000）
  - 含义：物理引擎中“接触对（contacts）”缓冲上限
  - 影响：
    - 显存：显著降低物理缓冲占用
    - 物理稳定性：过低会丢弃多余接触，可能出现脚穿透/打滑；若看到“接触对超限”/脚下不稳，需上调
  - 出处：`SIM_CFG` in velocity_env_cfg.py

### 模型保存位置
运行时的工作目录（CWD）不同，导致 logs/ 生成在“你当时启动命令的目录”下，而不是仓库根目录。

### 恢复到“最新一次保存”的模型（自动选择）
```bash
MUJOCO_GL=egl uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --agent.resume \
  --agent.experiment_name go1_velocity
```
含义：在 `logs/rsl_rl/go1_velocity/` 里选最新的 run 和其中最新的 `model_*.pt`。

### 用精确路径恢复（你的例子）
若你要从 `logs/rsl_rl/go1_velocity/2025-11-07_00-03-02/model_50.pt` 继续训练：
```bash
MUJOCO_GL=egl uv run train Mjlab-Velocity-Rough-Unitree-Go1 --agent.resume True --env.scene.num-envs 512
```
这会自动加载 logs/rsl_rl/<experiment_name>/ 下最新运行的最新检查点。

方式 2: 指定特定运行和检查点
```bash
MUJOCO_GL=egl uv run train Mjlab-Velocity-Rough-Unitree-Go1 \  
  --agent.resume True \  
  --agent.load-run "2024-11-09_15-30-45" \  
  --agent.load-checkpoint "model_5000.pt" \  
  --env.scene.num-envs 4096
  --agent.max-iterations 1 \
  --agent.save-interval 1
```
这会加载指定日期时间的运行中的特定检查点文件。
方式 3: 使用正则表达式匹配
```bash
MUJOCO_GL=egl uv run train Mjlab-Velocity-Rough-Unitree-Go1 \  
  --agent.resume True \  
  --agent.load-run "2024-11-09.*" \  
  --agent.load-checkpoint "model_[0-9]+.pt" \  
  --env.scene.num-envs 4096
```

### tensorboard
checkpoint只是权重快照，不能起到部署的作用
必须转化成合适的onnx文件才可以。平台里的 “play/评估” 之所以能直接用，是因为它在同一套 Python 训练代码中，能重建模型结构并套用同样的前后处理
uv run train Mjlab-Velocity-Rough-Unitree-Go1 \
  --env.scene.num-envs 1 \
  --agent.logger wandb \
  --agent.resume True \
  --agent.load-run '2025-11-08_*' \
  --agent.load-checkpoint 'model_1500.pt' \
  --agent.max-iterations 1 \
  --agent.save-interval 1

--enable-nan-guard True
uv run tensorboard --logdir logs/rsl_rl/go1_velocity/2025-11-12_00-27-58 --port 6006
## python 语法

```python
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg

#dataclass = 方便定义“配置对象”的类；字段就是类的属性
#自动为你生成常见的方法（init, repr, eq 等），方便把“配置/数据结构”声明成类而不用手写构造函数。

@dataclass
class LocomotionVelocityEnvCfg(ManagerBasedRlEnvCfg):
  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  # scene是实例的属性名，创建实例后可以使用cfg.scene访问
  # python中，创建实例是这样的 cfg = EnvCfg() 
```   
  scene: Scentcfg 的意思是
   ``` cfg.scene ```
  这个东西应该是一个 ``` SceneCfg ```的实例

ONNX metadata 的作用：让你不需要重建环境就能知道准确的观测项顺序、关节顺序、缩放等。使用 .pt 则没有这些附加键值。


在play的时候，会有这个表格：
[INFO] <TerminationManager> contains 3 active terms.
+------------------------------------+
|      Active Termination Terms      |
+-------+-----------------+----------+
| Index | Name            | Time Out |
+-------+-----------------+----------+
|   0   | time_out        |   True   |
|   1   | fell_over       |  False   |
|   2   | illegal_contact |  False   |
+-------+-----------------+----------+
Time Out：是否把该终止当作“超时”（Gym 的 truncated）而不是“失败终止”（terminated）
True 表示“超时”类（truncated），不算失败；用于固定时长的回合切分
False 表示“失败”类（terminated），通常意味着机器人状态异常（倒了、违规接触等）


```bash
python -m mjlab.scripts.play Mjlab-Velocity-Rough-Unitree-Go1-Play \
  --agent trained \
  --wandb-run-path <你的wandb_run_path 或加 --checkpoint-file /path/to/checkpoint.pt> \
  --num-envs 32 \
  --viewer native
  --agent.logger tensorboard
```
- 注意，加不加-Play非常关键，加了-Play之后是变成 -play的项目环境，是无限长度的play
  - 注册了一个“面向演示的变体配置注册了一个“面向演示的变体配置

Space（空格）: 暂停 / 恢复（toggle pause）
Enter: 重置环境（reset）
-（减号 / KEY_MINUS）: 减速（request_speed_down -> decrease_speed）
=（等号 / KEY_EQUAL；通常 Shift+= 为 +）: 加速（request_speed_up -> increase_speed）
,（逗号 / KEY_COMMA）: 切到上一个环境（PREV_ENV）
.（句点 / KEY_PERIOD）: 切到下一个环境（NEXT_ENV）
P（大写 P / KEY_P）: 切换/显示 reward plots（TOGGLE_PLOTS）
R（大写 R / KEY_R）: 切换/隐藏 Debug Visualization（TOGGLE_DEBUG_VIS）

## 地形生成

### 地形网格是“列=地形类型、行=难度”的二维表。
训练时并不是让机器人“跑到某处找台阶”，而是“把该并行环境的出生点直接放在某个网格格子上”。
调难度的方法主要是“切换它所在的行（terrain_level）”，台阶列里行数越大，台阶越高/更难。
网格与出生点
Procedural 生成器会构建一个 num_rows × num_cols 的地形网格：
列（col）由 sub_terrains 的 proportion 决定类型分配：flat、pyramid_stairs、pyramid_stairs_inv 等；
行（row）代表难度；curriculum 模式下，同一列从第 0 行到最后一行，difficulty 单调递增。
训练的每个并行环境 env_i 都有一个“出生点 env_origin_i”，它不是让机器人从世界中心走到某块地形，而是直接把它“放在对应格子的中心平台上”开始训练。
env_i 的当前位置由两组索引决定：
terrain_types[i] = 这个 env 使用哪一列（哪种地形类型）
terrain_levels[i] = 这个 env 使用哪一行（难度多少）
env.reset() 时会把该 env 的 origin 设置成 terrain_origins[level_i, type_i]。

### 生成过程


``` python
SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=5,
  ),
  num_envs=1,
  extent=2.0,
)   
```
#只是先导入相关数据 ，真正使用的时候是在 scene.py初始化时构造 TerrainImporter
``` TerrainGeneratorCfg```这个*数据类*是配置，他并非是执行端。
``` TerrainImporter ``` 读取 ``` TerrainGeneratorCfg ```的配置数据构造  ``` TerrainGenerator ```
TerrainGenerator会配置一系列关于地形的信息

### 添加 地形

## UV以及环境配置问题
echo "nameserver 8.8.8.8" | tee /etc/resolv.conf  解决DNS问题


## 安装
### 换源 url = "https://mirrors.nju.edu.cn/pytorch/whl/cu128/"