# 跑酷任务全面重构方案（以 wmp_go1 训练细节为蓝本）

> 参照仓库：`/home/mast/github_resources/wmp_go1`（WMP: World Model-based Perception,
> arXiv:2409.16784 改进版，Isaac Gym + Warp 深度射线，Go1 实机已验证）。
> 本文档基于对其 `legged_robot.py`（奖励/终止/课程实现）、`terrain.py`（地形生成）、
> `a1_amp_config.py`（全部超参）、`doc/TERRAIN_COMMANDS.md`、`doc/TERRAIN_PARAMETERS.md`
> 的逐行精读，重构 `src/mjlab/tasks/parkour_vision/`。
>
> 2026-10-15

---

## 0. 本轮失败复盘 → 重构动机

bonus=200 训练（run `2026-10-15_14-10-30`，4924 轮 NaN 崩溃）play 效果差。三个叠加原因：

1. **anti-stall 回归 bug**（我埋的）：墙列 vx≥0.15 时返回完整基础项（含 vz 惩罚），
   跳跃腾空瞬间 w=4 跟踪奖励整项归零——跳跃被重罚，比修复前更不翻墙；
2. **相机 30° 下俯盲区**：视线中心落地 ~1.1m，墙在 2.5~3.5m 处贴视野上缘，看不清；
3. **逐列开洞的奖励结构**：墙列豁免了 tracking-vz / upright / 接触惩罚 / 接触终止
   四处，每处豁免都是一个奖励梯度断崖，策略在洞边缘找到大量局部最优（绕行、蹭分）。

## 1. wmp_go1 的核心设计哲学（与我方对比）

| 维度 | wmp_go1 | 我方现状 | 重构方向 |
|---|---|---|---|
| 奖励结构 | **统一**：全部 17 种地形共用一套奖励，无任何按地形豁免 | 三列地形四处列级豁免 | 删除豁免洞，用**速度裁剪**替代 |
| 通过障碍的方式 | 所有障碍都"**保持速度冲过去**"（heightfield 无真垂直墙、浅坑可爬出、深渊坑用 feet_edge 引导防踩空） | 真垂直 box 墙，需停-爬-越 | 保留奖励点主驱动，但**超速不罚**让冲越有利 |
| 停顿处理 | **stuck 惩罚 -1.0**（指令>0.1 而 \|vx\|<0.1） | anti-stall 中性化（有 bug） | 改为 stuck 惩罚（反方向、无洞） |
| 速度指令 | forward 组 [0,0.5]→课程升至 [0,1.0]，按列组 tracking 分数 >0.8 每次升 0.2 | 静态两阶段（按迭代步） | 改为分数驱动的指令课程 |
| 里程碑 | 无奖励点 | reward_point_bonus | **保留**（任务定义，robot parkour 风格） |
| AMP | 判别器 + mocap | 无（第一阶段决定不变） | 不加 |

**一句话**：wmp_go1 证明"不需要给任何地形开洞"——把 tracking 做成
"实际速度 ≤ 指令+0.1 就满分"的裁剪形式，障碍段冲得快不受罚、停顿被 stuck 罚、
边缘踩空被 feet_edge 罚，一套奖励通吃所有地形。

## 2. wmp_go1 关键训练细节精读（精读笔记）

### 2.1 奖励（`legged_robot.py:1607-1785`，σ=0.25）

| 项 | 权重 | 实现要点 |
|---|---:|---|
| tracking_lin_vel | **1.0** | **lin_vel_clip=0.1**：仅对 gap/bream/continuous_gap/climb 四列，实际速度超过 指令±0.1 的部分被裁剪后再算误差——**超速冲障碍不受罚**；其余地形原始误差 |
| tracking_ang_vel | 0.5 | 常规 |
| lin_vel_z | **-1.0** | **无任何豁免**（vz² 直接罚，跳跃代价由任务结构承担） |
| feet_air_time | 1.0 | 目标 0.5s，仅首次触地结算，指令<0.1 不给 |
| feet_edge | **-1.0**（课程 0.1→1.0 @ 4000→10000 迭代） | 脚踩"高度边缘 mask"（沟壑沿 + 矮墙沿的悬崖线，`x_edge_mask` 膨胀 1 像素）即罚——**防踩空落坑的正则**；仅对 gap+climb 列生效 |
| feet_stumble | -1.0 | 脚水平接触力 > 4×垂直力（踢墙） |
| collision | -1.0 | thigh/calf 接触 >0.1N |
| cheat | -1.0 | forward 组：**世界系**朝向偏离 +x 超 ±1.0 rad 即罚（绕行的直接惩罚） |
| stuck | **-1.0** | 指令>0.1 且 \|vx\|<0.1（罚停顿，与 anti-stall 相反） |
| stool_base_height | -1.0 | 仅 stool 列：基座高于出生面 0.45m 罚（防跳凳顶作弊） |
| torques/action_rate/smoothness/dof_acc | -1e-4/-0.03/-0.01/-2.5e-7 | 常规平滑项 |

### 2.2 终止（`legged_robot.py:359-373`）

```
contact_terminate: base 接触 > 1N            （无任何地形豁免——heightfield 无真垂直墙，躯干不会被迫贴墙）
vel_violate:     倒退超 1.5 m/s，仅 level>3 生效  （防倒着跑刷分）
fall:            vz < -3.0 或 projected_gravity_z > 0（翻倒）
time_out
```
**没有**落坑高度终止——因为坑要么是深渊（触底即摔翻）要么浅坑 -0.3m（能爬出来继续拿分）。

### 2.3 地形课程（`legged_robot.py:940-955`）

- 升行：行走距离 > env_length/2（4m，与我方一致）；
- **降行**：距离 < 指令速度 × max_episode_length_s × 0.5（走不到一半就降级）；
- **满级回流**：到顶后随机分到任意行（持续复习简单地形）。

### 2.4 指令课程（`legged_robot.py:966-1000` + doc/TERRAIN_COMMANDS.md）

- forward 组（gap/continuous_gap/bream/stepping_*/climb/stool/crawl——**与我们三地形同组**）：
  - X ∈ [0, 当前上限]，Y=0，heading=0 固定；
  - 组内累计 tracking_lin_vel 分数（episode 归一化），**>0.8 时 X 上限 +0.2**（0.5→1.0 封顶）；
  - XY 合速 <0.2 清零；世界系 |yaw|>1.0 rad 吃 cheat 惩罚。

### 2.5 与我方三地形对应的地形参数（`terrain.py:298-495`）

| 我方地形 | wmp 对应 | wmp 参数 | 对我方的启示 |
|---|---|---|---|
| 沟壑 | `gap` | 缝 0.1→0.9m，坑 = **-5m 深渊**（无坑底，掉下必摔），配 feet_edge | 缝宽映射用归一化难度 i/9；深渊坑不需要掉坑终止 |
| 沟壑 | `continuous_gap` | 缝 0.08→0.5m，坑仅 **-0.3m 浅坑可爬出**，中间平台宽 0.3-0.4m、高随机 0-0.15m | 浅坑让"掉下去"不等于死，探索更友好 |
| 矮墙 | `climb` | 高 0.15→0.60m，**两组障碍**各长 0.9-1.1m、全宽、间隔 2.5-2.9m | 双墙链 + 全宽；heightfield 无真垂直墙（slope_threshold 0.75 斜面化）→ 可以冲上去，无需停顿 |
| 梅花桩 | `stepping_one_bridge` / `bream` | 桥段 0.2-0.5m、坑深 1m；bream 石块 0.5-0.9m、缝 0.1-0.4m、低矮(≤0.21m) | wmp 的石头全部贴地或低矮——我方梅花桩（高柱+深坑）更难，依赖奖励点引导 |

### 2.6 相机（`a1_amp_config.py:164-185`）

- 64×64、**FOV 58°**、near 0 / **far 2m**、update_interval=5；
- **pitch 15~25° 随机下俯**（每个 env 固定随机）+ yaw ±2°、roll ±2°、位置 ±1cm；
- 深度 clip 到 [-2,0] 取负归一化到 **[-0.5, 0.5]**。

**对比我方**：86° FOV / 4m / 固定 30° / scale 1/4。wmp 的 58° 窄视场 + 近量程 +
15-25° 随机俯仰是被实机验证过的配置——俯仰随机化天然覆盖"近处脚下"和"远处障碍"
两种视野，比固定大俯角盲区小得多。

### 2.7 PPO 细节（`a1_amp_config.py:257-303`）

- **`min_normalized_std = [0.05, 0.02, 0.05] × 4`（逐维 std 下限）**——正是我方
  NaN std 崩溃的正解：不是 clamp 到 1e-4，而是给每维一个有物理意义的下限；
- entropy 0.01、adaptive lr（与我方一致）；AMP 项第一阶段不用。

## 3. 重构方案（逐模块）

### 3.1 奖励层——删除全部豁免洞，换 WMP 双件套（核心改动）

文件：`src/mjlab/tasks/parkour_vision/mdp/rewards.py`、`parkour_env_cfg.py`

| # | 改动 | 具体内容 |
|---|---|---|
| R1 | **删除 anti-stall** | `track_linear_velocity_excluding_z` 整个函数废弃，恢复基础 `track_linear_velocity`，**三列统一、vz 惩罚统一**（对齐 WMP，无豁免） |
| R2 | **加 lin_vel_clip=0.1 速度裁剪** | 新增 `track_linear_velocity_clipped`：三列全部生效（我方全是障碍列）——实际 xy 速度超过 指令+0.1 的部分裁剪后再算 exp 误差（σ=0.25）。冲障碍/跳跃期速度偏快不罚 |
| R3 | **tracking 权重 4.0 → 1.0** | 对齐 WMP；σ 保持 0.25（继承值一致） |
| R4 | **bonus_value 200 → 100** | 重算量级：w=1 后满速 tracking ≈1.3/s（旧 w=4 时 1.3 是含 w 的值，裸 exp≈0.33/步×50=…）；单点入账 0.02×100=2.0 ≈ 2 秒满速跟踪收益；梅花 9 点=18.0、墙 2 点=4.0、沟 3 点=6.0。**墙列跳跃期 vz 归零的 tracking 损失（~1.0×0.3s）由 4.0 墙分覆盖** |
| R5 | **新增 stuck 惩罚 w=-1.0** | 指令>0.1 且 \|vx\|<0.1 → 每步罚。替代 anti-stall：停顿不再是"中性"而是负收益，与奖励点前进驱动同向 |
| R6 | **新增 cheat 惩罚 w=-1.0** | 世界系朝向偏离 +x 超 ±1.0 rad → 罚。与 heading_alignment(w=1) 叠加：alignment 给对齐正分，cheat 罚大偏航 |
| R7 | **新增 feet_stumble w=-1.0** | 脚水平接触力 > 4×垂直力 |
| R8 | **保留** reward_point_bonus / heading_alignment / lateral_track / forward_progress | forward_progress w 3.0→1.0（与 tracking 同量纲，避免位移奖励压过一切） |
| R9 | **保留墙列接触类豁免，但收窄范围** | WMP 无豁免的前提（heightfield 无真垂直墙）在我方不成立——真 box 墙攀爬贴墙不可避免。**保留** head/thigh/calf 接触惩罚豁免 + illegal_contact 终止豁免 + 85° 放宽倾角（用户此前明确要求）；**删除** upright 豁免（恢复统一 upright：攀爬前倾损失 ~0.87/步×w，量级可被墙分覆盖，且防攀爬时后仰刷 upright——若重训发现仍抑制攀爬，Phase 2 再回补） |

量级核对（w=1 体系）：满速直行 tracking≈1.0、forward_progress≈0.5/s(0.5m/s×1)、
air_time≈0.5/s、heading+lateral≈1.5/s → 正常行走 ~3/s；翻墙链 2 点=4.0 ≈ 1.5s
正常收益，攀爬期损失（vz+upright+stuck 风险）~2-3 → **净收益转正**，且绕行被
cheat/越界终止硬性掐断。

### 3.2 指令层——分数驱动课程（对齐 WMP forward 组）

文件：`parkour_env_cfg.py`、`mdp/commands.py`（或复用改造 command_vel curriculum）

| # | 改动 | 具体内容 |
|---|---|---|
| C1 | 初始指令范围 (0.3,1.0) → **(0.0, 0.5)** | WMP forward 组起步值；从走开始学 |
| C2 | 静态两阶段 → **分数驱动**：组内 tracking 分数 >0.8 时 X 上限 +0.2，封顶 1.0 | 复用现有 command_vel curriculum 挂钩 episode_sums["track_linear_velocity"]，按列组聚合（WMP `_accumulate_command_tracking_score` 逻辑，含"零步 reset 不计入"） |
| C3 | 合速 <0.2 清零门控 | 指令采样后 gate（WMP 同款，防学出原地抖动刷分） |
| C4 | 保留纯 x、yaw 固定 0 | 任务定义不变 |

### 3.3 终止层

文件：`src/mjlab/tasks/parkour_vision/mdp/terminations.py`

| # | 改动 | 具体内容 |
|---|---|---|
| T1 | 新增 **vel_violate** | 基座 x 速度比指令低超 1.5 m/s（倒退狂奔）且 terrain_level>3 → 终止（WMP 同款；补 out_of_bounds 的空隙——空间上没出界但方向反了） |
| T2 | fell_over 判据简化 | 保留 75°/85° 分列（攀爬需求），不改 |
| T3 | 保留 parkour_completed / out_of_bounds / fell_into_pit | 我方深坑（0.4-1.0m 有坑底）与 WMP 深渊/浅坑不同，落坑终止保留 |

### 3.4 地形层（保留用户定制规格，只调结构）

用户此前明确的规格（梅花缝 0-0.2、沟缝 0-0.5、墙高 0.2-0.5、坑深 0.4-1.0、桩宽 0.35）
**全部保留**。参考 wmp 的结构性调整：

| # | 改动 | 具体内容 |
|---|---|---|
| D1 | 难度行映射改归一化 i/9 | 缝隙类（沟壑、梅花）用 `i/(rows-1)` 全程 0→上限（WMP 对 gap 类的做法），避免最后一行永远到不了上限 |
| D2 | 矮墙改**双墙**（建议项，需确认） | 对齐 climb：两道墙各厚 0.4-1.2m、间隔 2.5-2.9m，每墙 2 奖励点。链结构与梅花（8 对）/沟壑（3 缝）一致，"通过感"更强。**若不改则维持单墙** |
| D3 | 沟壑中间平台高度随机 0-0.15m（小改） | 对齐 continuous_gap：三缝之间的平台段高度随机微抬，增强深度信号多样性 |

### 3.5 相机层（解决盲区，实机验证过的配置）

| # | 改动 | 具体内容 |
|---|---|---|
| V1 | 俯仰 **固定 30° → 15~25° 每 env 随机**（域随机化事件 reset 时采样） | WMP 同款；既有俯视脚下又有平视前方，策略对安装误差鲁棒。用户"总 30°"规格的初衷（多看地面）由 25° 上限+窄 FOV 近量程覆盖 |
| V2 | far_clip 4.0 → **3.0m**，深度归一化改 clip[-3,0]→取负→[-0.5,0.5] | 对齐 WMP 量纲（其实机验证配置）；旧 scale 1/4 在 4m 处分辨率差 |
| V3 | FOV 86° → **87° 保持** | 86° 已接近 D435；WMP 的 58° 是其 RSSM 架构选择，CNN 直连架构宽 FOV 更稳（不改） |

### 3.6 训练层

| # | 改动 | 具体内容 |
|---|---|---|
| P1 | **逐维 min_std**：GaussianDistribution 加 per-dim 下限 [0.05, 0.02]（动作前 6 维 hip/thigh 用 0.05、calf 0.02，或统一 0.02） | NaN 崩溃正解（WMP `min_normalized_std`）；替换现 clamp(1e-4)，nan_to_num 防护保留作最后兜底 |
| P2 | feet_edge 课程（Phase 2） | 若实现 feet_edge：权重 0.1→1.0 @ 4000→10000 迭代（WMP 同款） |

### 3.7 不改的清单（明确边界）

- 任务定义（纯 x 前进、直线障碍链、成功终止、越界终止）；
- reward points 机制本身（hit_radius 0.4 / z_margin 0.05 / 前进门控）；
- 三种地形规格参数（用户定制）；
- PPO 超参（adaptive lr、desired_kl 0.01、entropy 0.01）；
- 不加 AMP（第一阶段决定）；CNN 直连感知架构（不引入 RSSM 世界模型）。

## 4. 实施顺序与验证

| 阶段 | 内容 | 验证 |
|---|---|---|
| Phase 1（本次） | R1-R9 + C1-C4 + T1 + V1-V2 + P1 | 本地 `check_camera.py`（俯仰范围）、tyro 加载 smoke、短训 500 轮看 reward 曲线量级是否符合预期（tracking≈1.0、bonus 逐步出现、无 NaN std） |
| Phase 2 | feet_edge（需在 TerrainOutput 加边缘 mask）+ D2 双墙 + D3 | 短训验证后合入 |

Phase 1 从零训练（策略结构不变但奖励语义大改，旧 checkpoint 无续跑价值），
服务器 A800 ~5h/6000 轮。

## 5. 风险与回退

- tracking 4.0→1.0 + upright 豁免删除是最大行为变化：若短训发现攀爬仍被抑制，
  回补 upright 豁免（R9 预留）或将墙列 vz 权重减半——两处都是一行改动；
- 俯仰随机化会加大视觉分布宽度：早期收敛变慢属正常，6000 轮内应恢复；
- 全部改动集中在 `parkour_vision` 任务目录 + `distribution.py`，不触碰框架层。
