# 代码修改检查清单（Code Change Checklist）

> 只记录**每次修改必做**的全局要求。单任务/特定领域的细节不写入，保证简洁。

## 1. 修改纪律（改之前）
- [ ] 只改必要的地方：不移动/重排已有代码行，不重命名未涉及的变量
- [ ] 不碰无关文件；避免 `ruff --fix` 误删有用 import（必要时加 `# noqa: F401`）
- [ ] 最小 diff；注释语言与文件一致

## 2. 静态检查（必做）
- [ ] `uv run ruff check <改的文件>` — 无新增错误（既有错误不修，避免污染）
- [ ] `uv run ruff format <改的文件>` — 格式一致
- [ ] `uv run ty check` — 改到 rl/ 或新增类时
- [ ] `uv run python -c "from <模块> import <新符号>"` — 可导入

## 3. 最小冒烟测试（每次修改必做）
- [ ] 新函数/类：mock 环境构造最小调用，验证核心逻辑（正常分支 + 边界分支 + reset）
- [ ] 配置改动：确认 cfg 能取到新配置、类型正确
- [ ] obs 构建改动：确认维度/顺序符合预期

## 4. git 纪律
- [ ] 提交前 `git diff --stat` 确认只含预期文件
- [ ] 不提交 ruff 自动改动的噪音（format 全文件污染 diff）
- [ ] submodule（src/rsl_rl）改动单独提交，主仓库只更新指针

## 常见坑（踩过）
- `ruff --fix` 误删有用 import（F401）→ 加 `# noqa: F401` 或只用 `ruff check` 不用 `--fix`
- `scene.get("sensor")` 不存在，Scene 用 `scene["sensor"]`（索引）
- ONNX export 的 `int(tensor[i])` 是 data-dependent，torch.export 会报错
- 状态型 reward 必须处理 reset（`episode_length_buf==0`），否则跨 episode 残留
