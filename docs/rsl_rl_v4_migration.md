# RSL-RL 4.0 迁移指南与冲突解决文档

本文档详细记录了将代码库从 `rsl_rl` 旧版本（<4.0）迁移到 `rsl_rl` 4.0+ 版本时遇到的主要变更、冲突及解决方案。

## 1. 核心架构变更概述

`rsl_rl` 4.0 引入了重大架构重构，主要包括：
- **模型类变更**: `ActorCritic` 类被移除或重构，取而代之的是 `MLPModel` 作为基类，且 Actor 和 Critic 的网络构建逻辑更加分离。
- **配置结构**: 配置文件中的 `policy` 部分被拆分为 `actor` 和 `critic` 两个独立部分。
- **观测组命名**: 默认的观测组名称由 `policy` 变更为 `actor`。
- **构造函数签名**: 模型的 `__init__` 方法签名发生变化，不再接受 `num_actions` 作为显式参数，而是使用 `output_dim`。

---

## 2. 详细冲突与解决方案

### 2.1 配置文件结构变更

**冲突现象**:
运行训练时报错 `KeyError: 'actor'` 或 `Config attribute 'policy' not found`。

**原因**:
旧版本配置将 Actor 和 Critic 的参数统一放在 `class_name` 同级的 `policy` 字典中。新版本要求显式区分。

**解决方案**:
在 `rl_cfg.py` 中，将 `policy` 拆分为 `actor` 和 `critic`。

**旧代码**:
```python
policy = {
    "class_name": "ActorCritic",
    "activation": "elu",
    "actor_hidden_dims": [128, 128, 128],
    "critic_hidden_dims": [128, 128, 128],
    # ...
}
```

**新代码**:
```python
# 必须指定完整的类路径字符串，避免导入错误
class_name = "mjlab.tasks.velocity_vision.rl.modules:DepthActorCritic"

actor = {
    "activation": "elu",
    "hidden_dims": [128, 128, 128],  # 注意属性名变更：actor_hidden_dims -> hidden_dims
    # ...
}

critic = {
    "activation": "elu",
    "hidden_dims": [128, 128, 128],
    # ...
}
```

### 2.2 观测组 (Observation Group) 命名变更

**冲突现象**:
报错 `ValueError: Observation 'actor' ... not found`。

**原因**:
`rsl_rl` 4.0 的 PPO 算法默认寻找名为 `actor` 的观测组，而旧代码通常定义为 `policy`。

**解决方案**:
1. 修改环境配置 `velocity_env_cfg.py`:
   ```python
   # 旧
   self.observations.policy = ...
   self.policy_terms = ...

   # 新
   self.observations.actor = ...   # 如果是 policy group，改名为 actor
   self.actor_terms = ...
   ```
2. 修改 `runner.py` 或相关读取观测维度的代码:
   ```python
   # 确保读取的是 "actor" 而不是 "policy"
   obs_dim = cfg.observations.actor.history_length * ...
   ```

### 2.3 模型类继承与导入 (Module Inheritance)

**冲突现象**:
`ImportError: cannot import name 'ActorCritic' from 'rsl_rl.modules'`。

**原因**:
`rsl_rl.modules.ActorCritic` 已不存在。现在的基类是 `rsl_rl.models.MLPModel`。

**解决方案**:
在自定义模块 `modules.py` 中修改导入逻辑兼容性：

```python
try:
    from rsl_rl.modules import ActorCritic
except ImportError:
    from rsl_rl.models import MLPModel as ActorCritic
```

### 2.4 构造函数签名不匹配 (Signature Mismatch)

**冲突现象**:
`TypeError: MLPModel.__init__() got an unexpected keyword argument 'num_actions'`。

**原因**:
`MLPModel` 的 `__init__` 方法签名已变更为：
`(self, obs, obs_groups, obs_set, output_dim, ...)`。
它不再接受 `num_actions`。PPO 算法在实例化 Policy 时会传入这些新参数。

**解决方案**:
修改自定义 ActorCritic 类（如 `DepthActorCritic`）的 `__init__` 方法：

1. 显式接收 `obs_set` 和 `output_dim`。
2. 将 `output_dim` 映射给旧逻辑需要的 `num_actions`。
3. 在调用 `super().__init__` 之前，过滤掉 `MLPModel` 不支持的参数（如 `num_actions`）。

```python
def __init__(self, obs, obs_groups, obs_set="actor", output_dim=None, num_actions=None, **kwargs):
    # 兼容处理
    if output_dim is not None and num_actions is None:
        num_actions = output_dim
    
    # 构造传给基类的参数
    base_kwargs = {
        "obs": obs,
        "obs_groups": obs_groups,
        "obs_set": obs_set,
        "output_dim": num_actions, # 基类需要 output_dim
        # ... 其他基类支持的参数
    }
    
    # 调用基类
    super().__init__(**base_kwargs)
    
    # ... 后续自定义网络逻辑
```

### 2.5 域随机化 (Domain Randomization) API

**冲突现象**:
`AttributeError: module 'mjlab.utils.math' has no attribute 'randomize_field'`. (或其他 DR 相关错误)

**原因**:
旧的 `mdp.randomize_field` 辅助函数可能已废弃或不兼容。建议直接使用类属性赋值。

**解决方案**:
在 `velocity_env_cfg.py` 中，使用具体的随机化类：

```python
# 旧
# mdp.randomize_field(attr="friction", ...)

# 新
from omni.isaac.lab.envs import mdp # 或者 mjlab 的对应路径
# 使用具体的随机化器
friction = dr.geom_friction(params=...)
mass = dr.body_mass(params=...)
```

---

## 3. 检查清单

在迁移完成后，请检查以下几点：
1. [x] `rl_cfg.py` 中是否存在 `actor` 和 `critic` 字典。
2. [x] `velocity_env_cfg.py` 中是否定义了 `self.observations.actor`。
3. [x] 自定义 Policy 类的 `__init__` 是否正确处理了 `output_dim` 并过滤了 `kwargs`。
4. [x] `class_name` 是否使用了完整的 "module:class" 字符串格式。

如有其他报错，请参考 `rsl_rl` 官方仓库的示例代码或回退查看 `mjlab/docs` 中的相关说明。
