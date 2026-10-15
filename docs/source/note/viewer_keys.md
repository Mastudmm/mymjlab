# mjlab Native Viewer 键盘与鼠标操作说明

适用范围：`play.py --viewer native`（底层是 `mujoco.viewer.launch_passive`，
mjlab 通过 `key_callback` 注入了自己的按键处理，
源码：`src/mjlab/viewer/native/viewer.py` 的 `_safe_key_callback`）。

## mjlab 注入的按键（训练/回放控制）

| 按键 | 功能 | 说明 |
|---|---|---|
| `Enter` | 重置环境 | 等价于 env.reset()，reward 历史同时清空 |
| `Space` | 暂停 / 继续 | 仿真暂停后仍可拖动视角 |
| `-`（减号） | 仿真减速 | 速度倍率 1/32 ~ 1x，共 9 档 |
| `=`（等号） | 仿真加速 | 速度倍率最高 8x |
| `,`（逗号） | 上一个环境 | 单 env 视角切换 |
| `.`（句号） | 下一个环境 | 单 env 视角切换 |
| `→`（右方向键） | 单步推进 | 暂停状态下手动单步 |
| **`P`** | **切换实时曲线窗口** | 打开/关闭 reward 与关节状态的 matplotlib 实时绘图（`--live-plot` 相关；TOGGLE_PLOTS） |
| **`R`** | **切换 Debug 可视化** | 显示/隐藏传感器 debug 射线（深度相机射线、地形扫描射线等；TOGGLE_DEBUG_VIS） |
| **`A`** | **显示全部环境** | 切换"只看当前 env"与"显示所有 envs"（TOGGLE_SHOW_ALL_ENVS） |

> `P`（实时曲线）与 `R`（debug 可视化）来自上游 PR #698
> "Overhaul viewer controls and real-time playback"（2026-02-26），
> **不是 1.6 才有的新功能**——本仓库所有分支（1.2 基座）都包含。

## MuJoCo viewer 原生鼠标操作（渲染窗口内置）

| 操作 | 功能 |
|---|---|
| 左键拖拽 | 旋转相机 |
| 右键拖拽 / 中键拖拽 | 平移相机 |
| 滚轮 | 缩放 |
| 双击物体 | 选中该 body/geom（窗口底部显示其世界坐标） |
| 双击空白 | 取消选中，回到 free camera |
| `Esc` | 关闭窗口 |

## 如何查看 head site 的位置

1. **直接目视**：`head` site 是一个 `size=0.02` 的**红色小方块**（group=5，
   MuJoCo viewer 默认渲染 site），位于躯干前上方——相机安装点；
2. **精确读数**：双击红色小方块选中它，窗口底部 info 栏会显示其世界坐标
   （应为 `(0.3, 0, 0.2)` 相对躯干，世界 z ≈ 0.645）；
3. **数值脚本**（不启动 viewer）：

   ```bash
   uv run python src/mjlab/scripts/check_camera.py
   ```

   输出相机 forward 向量与俯仰/偏航角，用于核对安装规格
   （当前规格：头顶上方 15cm、视线朝 +x 下俯 15°）。

## 注意事项

- `R` 键关闭 debug 可视化后，深度相机射线束会消失——排查相机朝向时请保持开启；
- 暂停（`Space`）时射线束静止，适合逐帧观察相机视野覆盖；
- 多 env 模式下 `A` + `,`/`.` 组合可以快速定位某个 env 的相机视野。
