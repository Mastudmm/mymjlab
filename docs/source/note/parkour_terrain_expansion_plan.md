# AMP-Parkour 地形扩展方案（gap 跨缝 / 低趴过杆 / 原生台阶）

状态：**已定稿**（2026-11-01，§8 全部决策点经用户确认）。可开工实现。
对应任务：`Mjlab-ParkourVisionAMP-Unitree-Go1`（parkour_vision 任务层）。

---

## 0. 现状一句话

- 地形：`PARKOUR_TERRAINS_CFG` 只有 `platforms`（翻越高台）1 种 × 3 列，10 行难度阶梯（墙高 0→0.5 m）。
- 命令：`WaypointVelocityCommand`（命令名 `"twist"`，3 维 [vx, vy, wz]），恒定**航点模式**——每步指向下一个奖励点。
- 奖励：`reward_point_bonus` 按 `terrain.reward_points` + `reward_point_factors`（高 1.0 / 低 0.2）结算里程碑，落地门控 + 前进门控。
- 课程：`terrain_levels_by_pass_rate`（奖励点通过率升行）+ `commands_vel_score`（跟踪分提速）+ `waypoint_timeout_schedule`（通过率收紧限时）。

## 1. wmp_go1 参考实现对照（验证用户设想）

### 1.1 gap 跨缝地形
- wmp_go1 `gap_terrain`：**平地壕沟**（heightfield 挖穿，gap 0.10→0.90 m 随难度）；`continuous_gap_terrain`：平地多段平台 + 0.1→0.5 m 缝。都是**地面高度**，没有"先爬高台、再在高台顶上跨缝"的组合。
- 用户设想（爬上高台后紧跟同高台、缝 0→30 cm）是**新的组合地形**：以现 `BoxStraightPlatformTerrainCfg` 为几何底 + 把墙间平地换成穿到底的坑。
- 结论：设想成立，wmp_go1 只能借鉴"缝宽随难度插值"的量纲习惯；几何自研（见 §2.1）。

### 1.2 crawl 低趴过杆
- wmp_go1 `crawl`（terrain.py L254-270）：每 patch **2 根全宽悬空横杆**，杆底净高 `0.42 − 0.15·difficulty`（0.42→0.27 m），杆厚 0.2–0.5 m 随机，杆竖向 1.0 m（顶 ≥1.27 m，跳不过去只能趴下钻），位置 x≈4.0 与近末端。
- 结论：与用户"过横杆的低趴模式"一致，几何直接按 wmp_go1 量纲搬进 box-geom 风格（见 §2.2）。

### 1.3 command 与地形绑定（框架关键）
- wmp_go1 的做法：按 env 所在列把环境分 5 组（omni / rough_flat / stairs_discrete / forward / slope_cliff），**每组独立速度范围 + 独立课程**（`update_command_curriculum`）；forward 组命令纯 +x、heading 置零；全向组 heading 采样。
- mjlab 现状：`UniformVelocityCommandCfg` 已有 `terrain_command_ranges`（按地形名覆写采样范围）和 `heading_command` 模式（P 控朝向），但 `WaypointVelocityCommand` **没有**按地形切换命令模式的能力。
- 结论：需要补的核心框架件 = **单一命令项内部按列分发 waypoint / heading 两种模式**（见 §3）。

## 2. 三种新地形设计

### 2.1 `platform_gap`（高台 + 跨缝，bonus 5 点）
几何（沿 patch +x，全宽 7.4–7.5 m 防绕行；**2026-11-01 用户修订**：两台完全相同 + 跑道加深）：
```
spawn跑道(2.05-2.35m) → 高台1(h, 深1.2-1.6) → 缝g(0→0.3m, 穿到坑底) → 高台2(与高台1全同) → 地面 → 终点
```
- **两块平台大小形状完全一致**（用户修订 2026-11-01）：高度/深度/宽度全部共享同一 patch 的一次采样，不再独立采深度。
- `platform_depth_range=(1.2, 1.6)`（用户修订 2026-11-01）：台顶即加速区——爬上后到缝前有 ≥1.2 m 加速空间。
- `wall1_left_x_range=(2.8, 3.1)`（用户修订 2026-11-01）：出生到台1 的平地加速跑道 2.05–2.35 m（计 ±0.2 m 重置抖动最差 1.85 m）。预算核查：最坏 3.1+1.6+0.3+1.6 = right2 6.6 → 终点 7.6、成功线 8.0，与 platforms 同界。
- `platform_height_range=(0.0, 0.5)` + 抖动，与现 platforms 完全同一阶梯；两台**同高**。
- `gap_range=(0.0, 0.3)` 难度插值；缝为全宽坑（坑深 0.5–1.0 随机），掉进去由碰撞/倾倒终止或超时兜底（fell_into_pit 已取消注册）→ 跨缝必须是跳跃/大步跨越。
- reward points（factors：低=0.2，高=1.0，`max_reward_points=5`）：

| # | 位置 | z | 值 |
|---|------|---|----|
| 1 | 高台1 左缘 −0.30（上台前30cm） | 0 | 低 |
| 2 | 高台1 顶面中心（上台后） | h | 高 |
| 3 | 高台2 左缘 +0.25（跨gap之后） | h | 高 |
| 4 | 高台2 右缘 +0.30（下台后30cm） | 0 | 低 |
| 5 | 高台2 右缘 +1.00（终点，链尾） | 0 | 低 |

- 点 3 的捕获泡（hit_radius 0.125）必须完整落在高台2 顶面内：点右移到 `left2 + min(0.25, depth/2)`（自适配）。
- 链尾语义与现 platforms 一致：`parkour_completed` 在点 5 + 0.4 m 处结束回合。
- 每回合 bonus 总量 ≈ 2×1.0 + 3×0.2 = 2.6 单位（现 platforms = 3×1.0 + 4×0.2 = 3.8——2026-11-01 下台点由低值升级为里程碑，解决"爬上墙1就趴墙顶"的蹲营），同一量纲带，不打破 v8 调平的回报方差预算。

### 2.2 `crawl`（低趴过横杆，bonus 5 点——用户定版 2026-11-01：2 杆 + 3 低 2 高）
几何（wmp_go1 量纲，box-geom 风格）：
- 2 根全宽横杆（x 厚 0.2–0.5 m 随机，竖向 1.0 m，底缘净高 `clearance_range=(0.42, 0.27)` 难度插值），位置 x≈3.5–4.0 与 6.3–6.8；杆顶 ≥1.27 m 防跳越。
- 杆下净空就是通道：Go1 站姿躯干 ~0.3 m，难度 1 净高 0.27 m 必须趴低。
- reward points（提议，与 gap 地形同构的 3 低 2 高）：

| # | 位置 | z | 值 |
|---|------|---|----|
| 1 | 杆1 左缘 −0.30 | 0 | 低 |
| 2 | 杆1 右缘 +0.30（过杆1） | 0 | 高 |
| 3 | 两杆中间 | 0 | 低 |
| 4 | 杆2 右缘 +0.30（过杆2） | 0 | 高 |
| 5 | 杆2 右缘 +1.00（终点，链尾） | 0 | 低 |

- 复用 `reward_point_bonus`，**零新增奖励函数**；z 门控（base > 0.05）在地面自然满足。

### 2.3 原生台阶（正 / 逆，heading 命令）
- 在 `PARKOUR_TERRAINS_CFG` 增加两个原生子地形：
  - `BoxPyramidStairsTerrainCfg`：中心高台、四周下台阶（出生在中心 = **逆/下台阶**）；
  - `BoxInvertedPyramidStairsTerrainCfg`：中心坑底、四周上台阶（出生在中心 = **正/上台阶**）。
  - 已核实两者 origin 分别为 `+(num_steps+1)·step_height` / `−(num_steps+1)·step_height`，出生点都在 patch 中心。
- 参数（用户定版 2026-11-01，原生 mjlab 量纲）：`step_height_range=(0.01, 0.21)`（难度阶梯，1cm 起步）、`step_width_range=(0.3, 0.34)` + `step_width_mode="random"`（每 patch 独立采样）、`platform_width=3.0`（原生出生台）、`stair_lip_enabled=False`、`border_width=1.0`。极难度下金字塔顶/坑深 ≈ (5+1)×0.21 ≈ 1.26 m。
- 无 reward points（counts=0）→ 命令自动切 heading 模式；bonus 链、航点课程自然失效。
- 极难度下坑深见 §2.3 定版参数（≈1.26 m，原生量纲，用户已确认）。

## 3. 命令框架：单一 "twist" 命令项内部分发（核心改动）

**原则：不新增命令名、不改观测布局。** 观测 command 仍 3 维 [vx, vy, wz]，`track_goal_vel` / `variable_posture` / 步态门控 / AMP（43 维本体观测，不含命令）全部不动。

`waypoint_command.py` 扩展（类与 cfg 各加一段，不拆新文件）：
- cfg 新字段：
  - `heading_terrain_names: tuple[str, ...]`（如 `("pyramid_stairs", "pyramid_stairs_inv")`）；
  - `heading_speed_range: tuple[float, float] = (0.2, 0.5)`；
  - `heading_range: tuple[float, float] = (−π, +π)`（用户定版 2026-11-01：**全向**，与 mjlab 原生 heading 模式一致；patch 出入界由 `out_of_bounds` 的 ±3.5 m 兜底）。
- `__init__`：复制 `UniformVelocityCommand._resolve_curriculum_column_terrain_names` 的列分配算法，得到 列→地形名 映射；按列打 `is_heading_env` 掩码。
- `_resample_command`：heading env 额外采 `heading_target`（每回合一次）+ speed（`heading_speed_range`，写进同一 `speed` 缓冲，供 `commands_vel_score` 归一化）；waypoint env 走原逻辑。
- `_compute_direction_command`：waypoint env 原方向命令；heading env：`vx=speed, vy=0, wz=clip(k·wrap_to_pi(heading_target − heading_w))`（与 `UniformVelocityCommand._update_command` 同式）。
- `_advance_waypoints` / metrics：counts=0 的 env 循环条件天然 no-op（已逐行核对），只需 debug_vis 分流：heading env 画朝向箭头，waypoint env 画航点球。

备选方案（否决）：两个命令项并行 + 观测拼接——观测维度变化破坏续训，奖励/课程全部要加 env 掩码，复杂度高。

### 3.1 出生地形选择：采样与布局解耦（用户需求 2026-11-01）

需求：地形网格保持 5 列每类一列，但**机器人出生在哪个地形上要可选择**——训练时可选子集（如先只训 platforms+gap），play 时可选单列评估，且采样与网格布局无关。

代码事实（terrain_entity.py L329-348）：`terrain_types` 在启动时按 env 序号**轮流均分到各列**（`env_id // (num_envs/num_cols)`），此后终身不变（`update_env_origins` 只改行不改列）。因此"选择出生地形" = 启动分配这一处过滤即可覆盖训练与 play 全部场景。

实现：
- `TerrainEntityCfg` 新字段 `spawn_terrain_names: tuple[str, ...] | None = None`（None = 全部列，行为与现状逐位一致）。
- 地形名→列的解析目前散在 `UniformVelocityCommand._resolve_curriculum_column_terrain_names`（复制了生成器的分配算法）；抽成 `terrain_generator.py` 的模块级函数 `resolve_curriculum_column_names(generator_cfg)`，命令与实体共用（分配公式只写一遍，防三处漂移）。
- `_compute_env_origins_curriculum` 与 `randomize_env_origins`：若设置了 `spawn_terrain_names`，`terrain_types` 从**允许列集合中均匀随机采样**（替换轮流均分）；`terrain_levels` 逻辑不变。
- 用法：训练 cfg 可设子集；play cfg（`config/go1/__init__.py` 的 `play_env_cfg`）可设单列，实现"指定地形评估"。

## 4. 课程与终止的配套修改（含两个预挖出的 bug）

1. **`parkour_completed` 必须加 counts>0 门控**：无奖励点的列（台阶）上 `last_point = (0,0,0)`（预分配数组未填充的世界坐标），成功判定退化成"机器人是否在世界原点附近"——patch 恰好铺在世界原点附近的 env **出生即被判定通关重置**（网格以原点居中，第 5 行 patch x 从 0 起、第 1 列 y 横跨 0，两者同时命中的 patch 全程误触发），其余 patch 则永远不触发。两种情况都是错的：加一行 `valid = counts > 0` 修掉（用户已确认此修改，2026-11-01）。
2. **`fell_into_pit` 取消注册（用户决定 2026-11-01）**：不再用豁免方案。整体删除三处——`parkour_env_cfg.py` 的 `fell_into_pit` 终止项、`terminations.py` 的 `base_height_below` 函数、`mdp/__init__.py` 的导出（AGENTS.md 纪律：不留死代码）。已核实三者均无其他引用。
   - 对当前 platforms 地形：无坑，该项几乎不触发（仅极端低趴姿态 base<0.15 且未超 85° 倾角、躯干接触 <5N 的边缘情况），取消后行为近似不变。
   - 顺带消灭了逆台阶出生 bug（origin z≈−1.1m < 0.15 会在出生瞬间误杀）——这正是它"不需要"的真正原因。
   - 对新 gap 地形的代价：掉进跨缝坑的机器人不再立即重置，改由 `illegal_contact`（撞坑壁/坑底 >5N）、`fell_over`（30cm 窄缝内倾倒）或 20s 超时兜底；若训练中出现"坑内罚坐"现象，再为该列补一个按列门控的坑底检测（后备项）。
3. **`terrain_levels_by_pass_rate` 拆两组（用户已确认 2026-11-01）**：counts>0（bonus 三列）走通过率阶梯（原逻辑）；counts==0（台阶列）改走**距离阶梯**——离出生原点 > 2.5 m 升行、只升不降（对齐 velocity 任务 `terrain_levels_vel` 思想；不降级避免惩罚慢速爬台阶）。否则台阶 env 通过率恒 0 → 被一路降到 0 行，台阶难度永远上不去。
4. **`waypoint_timeout_schedule` 累计器只算 counts>0 env**：否则台阶 env 的 0 通过率稀释信号，限时永远收不紧。
5. **`commands_vel_score` 累计器只算 waypoint env**：台阶速度是静态范围（不参与全局提速），混算会稀释 waypoint 组的跟踪分（wmp_go1 分组课程的动机即此）。

## 5. 奖励侧冲突自查（逐项对照现 reward 表）

| 项 | 结论 |
|----|------|
| `reward_point_bonus` | 新地形只换点坐标与 factors，无新函数、无逻辑冲突 |
| `base_vz_l2`（−0.75） | 跨缝 vz≈0.5–1.0 约 0.3 s → 罚 ~0.05–0.1，≪ 8 分/点 bonus，与攀爬蹬伸同量级，不伤跳跃 |
| `air_time_over_penalty`（0.5 s 阈值） | 30 cm 缝滞空 ~0.3–0.4 s < 阈值 ✔；bonus 落地门控要求触地 → 必须落在缝对侧结算 ✔ |
| `upright` / `pose` / AMP 风格项 | 轻度偏爱站姿：45° 容差、std_walking 宽松、`amp_task_reward_lerp=0.95` 任务主导——量级核算均为软偏置，不阻塞低趴 |
| AMP 数据集（walk/back/turnleft/turnright） | **无匍匐动作**：风格奖励会给低趴帧打低分，但占比 ~5%×0.1，风险低；后续可补匍匐动作数据（决策点外，单独事项） |
| `head/thigh/calf_collision` | 蹭杆按力度扣分，天然适用 ✔ |
| `illegal_contact`（躯干 >5 N） | 躯干撞杆 = 判死（把"蹭杆"当失败反馈）。若训练中发现贴杆滑过太苛刻，改列豁免或降权重（训练观察项） |

## 6. 文件落点与改动量

| 文件 | 改动 |
|------|------|
| `src/mjlab/terrains/terrain_generator.py` | +模块级 `resolve_curriculum_column_names()`（列分配公式唯一化，供命令/实体共用） |
| `src/mjlab/terrains/terrain_entity.py` | `TerrainEntityCfg.spawn_terrain_names` + 两条 origin 分配路径的过滤（§3.1） |
| `src/mjlab/terrains/parkour_terrains.py` | +`BoxStraightPlatformGapTerrainCfg`、+`BoxStraightCrawlBarsTerrainCfg`（各 ~120 行，风格对齐现文件） |
| `src/mjlab/terrains/config.py` | `PARKOUR_TERRAINS_CFG`：num_cols 3→5，5 个 sub_terrains 等比（每列一种地形），参数全部显式 |
| `src/mjlab/tasks/parkour_vision/mdp/waypoint_command.py` | heading 分发（§3） |
| `src/mjlab/tasks/parkour_vision/mdp/curriculums.py` | 两组课程（§4.3–4.5） |
| `src/mjlab/tasks/parkour_vision/mdp/terminations.py` | `parkour_completed` counts>0 门控；**删除** `base_height_below`（fell_into_pit 取消注册） |
| `src/mjlab/tasks/parkour_vision/mdp/__init__.py` | 删除 `base_height_below` 导出 |
| `src/mjlab/tasks/parkour_vision/parkour_env_cfg.py` | `commands["twist"]` 新字段；terminations 表**移除** `fell_into_pit` 项；奖励表不动 |
| `tests/test_parkour_terrains.py` | +2 个地形的生成断言（点数/坐标/缝宽插值/杆净高插值） |
| `docs/source/changelog.rst` | Added 条目 |

## 7. 验证计划（冒烟）

0. **现状一致性验证（用户核心关切）**：保持 `PARKOUR_TERRAINS_CFG` 为旧 3 列 platforms 配置，用新代码构建 env：断言命令/终止/课程三张表与改动前逐项一致（除 `fell_into_pit` 按决定移除）、heading 掩码全 False、`parkour_completed` 门控全 True、跑 200 步对比 reset 统计无差异。
1. `uv run pytest tests/test_parkour_terrains.py`（新地形几何断言）。
2. 用 `scripts/tools/render_parkour_platform.py` 改造出 3 张新地形渲染图，人工核对几何与奖励点位置。
3. `uv run ty check`、`uv run pyright`、`uv run ruff format && uv run ruff check --fix`。
4. 最小训练冒烟：512 env × ~50 iter，确认：台阶列命令非零且 wz 为朝向 P 控制、台阶列无 bonus 结算、`parkour_completed` 零误触发、crawl 列基座高度分布（验证 ≥0.18 m，不碰 `fell_into_pit` 的 0.15 阈值）、五列课程独立推进。
5. play + debug 可视化：航点球（bonus 列）vs 朝向箭头（台阶列）分流正确。

## 8. 决策点——全部已定（2026-11-01 用户确认）

1. **gap 缝形式**：穿到底的坑（强制跳跃；掉落由 illegal_contact / 倾倒 / 超时兜底——fell_into_pit 已取消注册）。
2. **课程分组**：bonus 列通过率阶梯；台阶列距离阶梯（>2.5 m 升行、只升不降）。
3. **fell_into_pit**：整体取消注册（含 `base_height_below` 函数与导出）。
4. **crawl 点位**：2 杆 + 3 低 2 高。
5. **crawl 净高**：0.42 → 0.27 m（wmp_go1 量纲）。
6. **台阶 heading 范围**：±180° 全向（原生默认）。
7. **列分配**：5 列每类 1 列；另加 `spawn_terrain_names` 出生地形选择（§3.1）。
8. **台阶参数**：原生 mjlab 量纲——步高 (0.01, 0.21)、步宽 0.3–0.34 随机、platform_width 3.0。
9. **训练策略**：新配置**从零训**（提交训练前另行等用户明确指示）。
