import dataclasses
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from typing import Any

@dataclasses.dataclass
class ActorCfg(RslRlModelCfg):
    stochastic: bool = True
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"

cfg = RslRlOnPolicyRunnerCfg(actor=ActorCfg(distribution_cfg=None))
import json
print(dataclasses.asdict(cfg)["actor"])
