import torch
import torch.nn as nn

try:
    # Try importing ActorCritic from rsl_rl (works for versions < 4.0)
    from rsl_rl.modules import ActorCritic
except ImportError:
    # For rsl_rl >= 4.0, attempt to use MLPModel or define a placeholder
    try:
        # PPO usually expects a policy that behaves like ActorCritic.
        # In rsl-rl 4.0, MLPModel is the base for MLP policies.
        from rsl_rl.models import MLPModel as ActorCritic
    except ImportError:
        # Fallback to nn.Module if both fail (e.g. structure changed significantly)
        # This allows other tasks to load even if vision task is broken
        class ActorCritic(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
            def get_actor_obs(self, obs): return obs
            def get_critic_obs(self, obs): return obs

def get_activation_fn(activation_name):
    """
    根据传入的字符串名称返回对应的 PyTorch 激活函数。
    由于原始 rsl_rl 的 get_activation 导入可能存在兼容问题，此处本地实现。
    """
    if activation_name == "elu":
        return nn.ELU()
    elif activation_name == "selu":
        return nn.SELU()
    elif activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "crelu":
        return nn.ReLU()  # specific to rsl_rl, fallback
    elif activation_name == "lrelu":
        return nn.LeakyReLU()
    elif activation_name == "tanh":
        return nn.Tanh()
    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print(f"Activation function {activation_name} not found, defaulting to ELU")
        return nn.ELU()

class DepthActorCritic(ActorCritic):
    def __init__(
        self,
        obs,
        obs_groups,
        obs_set="actor",  # New in RSL-RL 4.0: typically "actor" or "critic"
        output_dim=None,  # New in RSL-RL 4.0: typically num_actions or 1
        num_actions=None, # Legacy support
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        # Handle positional args passed by RSL-RL 4.0: (obs, obs_groups, obs_set, output_dim, ...)
        # If output_dim is provided by PPO but num_actions is None, assume output_dim is num_actions
        if output_dim is not None and num_actions is None:
            num_actions = output_dim
        
        # Ensure num_actions is int
        if num_actions is None:
             raise ValueError("num_actions cannot be None. Please ensure output_dim or num_actions is passed.")
        
        # ----------------------------------------------------------------------
        # 1. Configuration & Dimension Calculation
        #    配置与维度计算
        # ----------------------------------------------------------------------
        # 从 kwargs 获取自定义参数
        # 注意: 这里的 depth_shape 如果是通过外部注册的参数传递的，比如 (1, 60, 80)，
        # 它会自动适配下方的 CNN 张量还原 (views)。但 flatten_dim 还是需要你在下面手动修改。
        self.depth_shape = kwargs.pop("depth_shape", (1, 80, 80)) # 默认C, H, W(1, 80, 80)，可由外部传入
        self.depth_history_num = kwargs.pop("depth_history_num", 1) # CNN处理多帧深度图 (本例中为1帧)
        
        # history_length 用于扩大除了展平的 depth 或 scan 外，其余本体信息的大小
        self.obs_history_num = kwargs.pop("obs_history_num", 10)  
        depth_vol = self.depth_shape[0] * self.depth_shape[1] * self.depth_shape[2]
        
        # 按照 rsl_rl 的模式，从传入的字典和观测组实时解包出所有的特征大小
        '''groups obs字典内拿到的是已经展平好的'''
        self.obs_groups = obs_groups  
        num_actor_obs = 0
        for obs_group in obs_groups["actor"]:
            num_actor_obs += obs[obs_group].shape[-1] 
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            num_critic_obs += obs[obs_group].shape[-1] 

        
        #对于 Actor来说
        # 我们假设 actor 的总观测 = 本体系特征10帧 + 深度图展平(depth_vol * depth_history_num帧)
        self.depth_total_vol_actor = depth_vol * self.depth_history_num
        self.proprio_total_dim = num_actor_obs - self.depth_total_vol_actor
        self.proprio_single_dim = self.proprio_total_dim // self.obs_history_num # 恢复出当前这一帧的特征维度

        # Scan Config (for Teacher/Critic)
        # 对于 Critic：已知总观测 = Critic特有本体与特权状态(非scan) + 地形扫描 (scan * 1帧)
        # Critic 不再使用 10 帧历史，所以此时 history 对于 critic 为 1
        self.scan_dim = kwargs.pop("scan_dim", 256) # 默认 256，允许外部传入避免硬编码
        self.scan_total_dim = self.scan_dim  #scan木有历史数据，直接等于一帧大小
        self.critic_proprio_priv_total_dim = num_critic_obs - self.scan_total_dim

        print(f"[DepthActorCritic] Architecture Dimensions:")
        print(f"  -> Actor Proprio Total: {self.proprio_total_dim}, Single Frame: {self.proprio_single_dim} | Depth Vol (Single): {depth_vol}")
        print(f"  -> Critic Privileged+Proprio Total: {self.critic_proprio_priv_total_dim} | Scan Base: {self.scan_dim}")

        # Latent Feature Dimensions
        # 隐层特征空间的表示维度
        self.visual_latent_dim = 32 # 由CNN生成的视觉隐特征维度输入进actor
        self.history_latent_dim = 32 # 由MLP提取的本体历史记忆隐特征输入进actor
        self.scan_latent_dim = 32   # 由Critic地形扫描压缩出的特征维度
        '''
        self.gru_hidden_dim = 512   # 时序核心GRU中隐藏状态向量的维度 (已弃用)
        '''

        # ----------------------------------------------------------------------
        # 2. Network Construction
        # ----------------------------------------------------------------------

        # RSL-RL 4.0 MLPModel signature:
        # __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=..., activation=..., obs_normalization=..., stochastic=...)
        
        base_kwargs = {
            "obs": obs,
            "obs_groups": obs_groups,
            "obs_set": obs_set,
            "output_dim": num_actions,
            "hidden_dims": actor_hidden_dims,  # Just to pass validation
            "activation": activation,
            "stochastic": True,
            "init_noise_std": init_noise_std,
        }
        
        # Filter kwargs for base class
        # MLPModel might accept: obs_normalization, noise_std_type, state_dependent_std
        valid_base_args = ["obs_normalization", "noise_std_type", "state_dependent_std"]
        for k in valid_base_args:
             if k in kwargs:
                 base_kwargs[k] = kwargs[k]
        
        # super().__init__ which is MLPModel.__init__ in RSL-RL 4.0
        # ...existing code...
        super().__init__(**base_kwargs)

        # Immediate overwrite of self.actor and self.critic
        activation_fn = get_activation_fn(activation)

        # ---------------------------
        # A. Visual Encoder (CNN)
        #    视觉编码器 (卷积神经网络) - 时序多帧独立卷积版
        #    作用: 将多帧带有噪声的深度图像分别送入同一个CNN提取每帧特征，随后经过MLP拼接压缩为定长的32维特征
        # ---------------------------
        # 修改深度图分辨率时，必须在此处手动计算并修改 flatten_dim。
        # O = floor( (I - K + 2*P) / S ) + 1
        # 当前默认输入图片: C=1, H=80(高度), W=80(宽度)   --若为80x60，则H=60, W=80
        # 
        # [层1] Conv2d(kernel=5, stride=1, padding=0): 
        #   - 对于 80x80: H -> 80-5+1=76, W -> 80-5+1=76.  输出形状: (32, 76, 76)
        #   - 对于 60x80: H -> 60-5+1=56, W -> 80-5+1=76.  输出形状: (32, 56, 76)
        #
        # [层2] MaxPool2d(kernel=2, stride=2, padding=0):
        #   - 对于 80x80: H -> 76/2=38, W -> 76/2=38.      输出形状: (32, 38, 38)
        #   - 对于 60x80: H -> 56/2=28, W -> 76/2=38.      输出形状: (32, 28, 38)
        #
        # [层3] Conv2d(kernel=3, stride=1, padding=0):
        #   - 对于 80x80: H -> 38-3+1=36, W -> 38-3+1=36.  输出形状: (64, 36, 36)
        #   - 对于 60x80: H -> 28-3+1=26, W -> 38-3+1=36.  输出形状: (64, 26, 36)
        #
        # [展平层] Flatten():
        #   - 对于 80x80: flatten_dim = 64 * 36 * 36 = 82944
        #   - 对于 60x80: flatten_dim = 64 * 26 * 36 = 59904
        #   - 对于 80×50（W × H） flatten_dim = 64 * 21 * 36
        flatten_dim = 64 * 36 * 36
        
        # 计算每一帧图像产出的向量维度，比如 n=1 就是 128; n=4 时就是 32
        self.per_frame_latent_dim = max(1, 128 // self.depth_history_num)
            
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(self.depth_shape[0], 32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            activation_fn,
            nn.Conv2d(32, 64, kernel_size=3),
            activation_fn,
            nn.Flatten(),
            nn.Linear(flatten_dim, self.per_frame_latent_dim),
            activation_fn
        )
        
        # 对全部 n 帧的独立特征进行全连接压缩
        self.visual_mlp = nn.Sequential(
            nn.Linear(self.per_frame_latent_dim * self.depth_history_num, self.visual_latent_dim)
        )

        # ---------------------------
        # B. Temporal History Encoder (MLP 代替 GRU 提取 10 帧 proprio)
        # ---------------------------
        self.history_encoder = nn.Sequential(
            nn.Linear(self.proprio_total_dim, 128),
            activation_fn,
            nn.Linear(128, self.history_latent_dim),
            activation_fn
        )
        
        '''
        # ---------------------------
        # C. Fusion & GRU (Temporal) - 【弃用】
        # ---------------------------
        # Fusion MLP: Combines Proprio + Visual Latent -> GRU Input
        fusion_input_dim = self.proprio_total_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            activation_fn,
            nn.Linear(128, 32),
            activation_fn
        )
        
        # GRU: Input 32 -> Hidden 512
        # GRU 网络作用: 为机器人提供短期的时序记忆，解决视觉被遮挡或动作间的时序依赖问题
        # 输入: 融合层输出特征, 形状 (Batch, Seq_len=1, gru_input_size=32)
        # 输出: (此步输出隐状态, 传给下一步记忆的状态), 最终提供隐藏维度 512 的结果
        self.gru_input_size = 32 + self.visual_latent_dim
        self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=self.gru_hidden_dim, batch_first=True)
        
        # Decoder: Hidden 512 -> Latent 32 (to be fed to Actor)
        # GRU 解码器作用: 将 GRU 庞大 (512维度) 的主导隐状态精炼还原为短小的特征(32维)，喂给最终的主 Actor 网络
        # 输入: GRU的主输出, 维度为 512
        # 输出: 供策略下达判断的提纯特征, 维度 32
        self.gru_decoder = nn.Linear(self.gru_hidden_dim, 32)
        '''

        # ---------------------------
        # C. Scan Encoder (Critic)
        #    扫描信息编码器 (Critic 专属)
        #    作用: 针对老师训练框架，利用全局准确的特权点云信息提升 Value 指导正确率
        #    输入: 地形精确一维测距扫描特征, 长度 scan_dim 
        #    输出: 一元特权隐特征, 维度 32
        # ---------------------------
        if self.scan_dim > 0:
            self.scan_encoder = nn.Sequential(
                nn.Linear(self.scan_dim, 256),
                activation_fn,
                nn.Linear(256, 128),
                activation_fn,
                nn.Linear(128, self.scan_latent_dim)
            )
        '''
        拟合scan dots的特权信息，输出32维度
        scan加上本体信息加上特权信息展平放入value network（512x256x128x1）里进行训练。
        '''
        # ---------------------------
        # D. Main Actor MLP
        #    主角策略网络 (Main Actor)
        #    作用: 根据机器人内部的本体状态 + 时序上的长线视觉记忆综合制定关节的目标指令
        #    输入: 拼接后的 [本体信息张量 (proprio_dim * obs_history_num), GRU输出提取张量 (32)], 所以总大小=proprio_dim * history + 32
        #    输出: 需要控制的所有动作维度 logits 预测 (num_actions)
        # ---------------------------
        # Input: Proprio + GRU_Decoder_Out (32)
        actor_in_dim = self.proprio_single_dim + self.history_latent_dim + self.visual_latent_dim
        
        actor_layers = []
        actor_layers.append(nn.Linear(actor_in_dim, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for i in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
            actor_layers.append(activation_fn)
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        self.actor = nn.Sequential(*actor_layers)
        
        # ---------------------------
        # E. Main Critic MLP
        #    价值评估网络 (Main Critic)
        #    作用: 基于准确不带噪声的状态，对未来的汇报/Value做出精准估算
        #    输入: 拼接后的 [本体信息 (proprio_dim * history), 压缩后特权地形点云扫描 (32)]
        #    输出: 单个标量期望回报 (Value), 维度 1
        # ---------------------------
        # Input: Proprio + Scan_Latent (32)
        # Or if scan_dim is 0, just privileged proprio
        critic_in_dim = self.critic_proprio_priv_total_dim + (self.scan_latent_dim if self.scan_dim > 0 else 0)
        
        critic_layers = []
        critic_layers.append(nn.Linear(critic_in_dim, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
            critic_layers.append(activation_fn)
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

        # Init weights
        self._init_weights()

    def load_state_dict(self, state_dict, strict=True):
        """
        拦截并修复被 runner.py 强行搬移过的 std 键名，
        使自定义模型能正确加载权重。
        """
        # 如果 runner.py 完成了迁移，但我们的模型依然使用旧的键名结构
        if "distribution.std_param" in state_dict and "std" in self.state_dict():
            state_dict["std"] = state_dict.pop("distribution.std_param")
        if "distribution.log_std_param" in state_dict and "log_std" in self.state_dict():
            state_dict["log_std"] = state_dict.pop("distribution.log_std_param")

        # 处理 visual_encoder (老名称) 到 shared_cnn + visual_mlp (新名称) 的自动兼容
        # 适用于旧 checkpoint 加载时。如果我们在跑单帧推理 (depth_history_num = 1)，权重可直接沿用
        if "visual_encoder.0.weight" in state_dict and "shared_cnn.0.weight" in self.state_dict():
            print("[DepthActorCritic] 正在将老版本 checkpoint 的 visual_encoder 迁移至 shared_cnn...")
            state_dict["shared_cnn.0.weight"] = state_dict.pop("visual_encoder.0.weight")
            state_dict["shared_cnn.0.bias"]   = state_dict.pop("visual_encoder.0.bias")
            state_dict["shared_cnn.3.weight"] = state_dict.pop("visual_encoder.3.weight")
            state_dict["shared_cnn.3.bias"]   = state_dict.pop("visual_encoder.3.bias")
            state_dict["shared_cnn.6.weight"] = state_dict.pop("visual_encoder.6.weight")
            state_dict["shared_cnn.6.bias"]   = state_dict.pop("visual_encoder.6.bias")
            
            # visual_encoder 的最后一步是 Linear(128, 32)，在新代码中它就是 visual_mlp 的第一步
            state_dict["visual_mlp.0.weight"] = state_dict.pop("visual_encoder.8.weight")
            state_dict["visual_mlp.0.bias"]   = state_dict.pop("visual_encoder.8.bias")
            
        return super().load_state_dict(state_dict, strict=strict)
        
    def _init_weights(self):
        # Orthogonal init
        for m in [self.actor, self.critic, self.shared_cnn, self.visual_mlp, self.history_encoder]:
            if isinstance(m, nn.Sequential) or isinstance(m, nn.Linear):
                if isinstance(m, nn.Sequential):
                    for layer in m:
                        if isinstance(layer, nn.Linear):
                            nn.init.orthogonal_(layer.weight, gain=2**0.5)
                            if layer.bias is not None: nn.init.constant_(layer.bias, 0.0)
                        elif isinstance(layer, nn.Conv2d):
                             nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=2**0.5)
                    if m.bias is not None: nn.init.constant_(m.bias, 0.0)

        if hasattr(self, 'scan_encoder'):
             for layer in self.scan_encoder:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=2**0.5)

    def reset(self, dones=None):
        # 负责处理循环或记忆模型的隐藏状态
        # 当环境 (env) episode 结束重置时，清空对应位置的隐藏记忆
        '''
        if hasattr(self, 'actor_hidden_state') and self.actor_hidden_state is not None:
            if dones is None:
                self.actor_hidden_state.fill_(0.0)
            else:
                self.actor_hidden_state[:, dones, :] = 0.0
        '''
        pass

    def _process_actor_obs(self, obs):
        # 处理 Actor 的实际包含图像观测的数据流，进行前向传播
        proprio = obs[:, :self.proprio_total_dim]
        # 由于 policy 设定了 proprio 的 history_length = 10，这里的 depth 已经是设置了单帧长度或外部设定长度（即只有 depth_history_num 个帧）
        depth_all_frames = obs[:, self.proprio_total_dim:]
        
        # 1. 截取最新单帧本体观测 (当前状态 obs)
        curr_proprio = proprio[:, -self.proprio_single_dim:]
        
        # 2. 本体历史通过 MLP 提取特征
        history_latent = self.history_encoder(proprio)

        # 3. 读取设定的最新深度的信息帧张量
        depth_vol = self.depth_shape[0] * self.depth_shape[1] * self.depth_shape[2]
        latest_depth_frames_dim = depth_vol * self.depth_history_num #这两行没用到吧


        depth = depth_all_frames

        # 把一维展平的全部深度图帧拆分为一帧一帧的张量，送回卷积层
        B = obs.shape[0]
        n_frames = self.depth_history_num
        C, H, W = self.depth_shape
        
        # 展平 -> (Batch * n_frames, Channels, Height, Width)
        # 这样网络会将 Batch 和帧数统一视为样本量来进行平行独立的 CNN 计算
        depth = depth.view(B * n_frames, C, H, W) 
        
        # Shared CNN: 利用同一套权重分别提取每一帧的隐特征
        # 输出: (B * n_frames, per_frame_latent_dim)
        frame_features = self.shared_cnn(depth)
        
        # 将 n 帧的数据重新放回 Batch 的平行维度中
        # 输出: (Batch, n_frames * per_frame_latent_dim)
        frame_features = frame_features.view(B, n_frames * self.per_frame_latent_dim)
        
        # 用 MLP 压缩出给到最终网络用于结合体感状态的 32 维特征
        visual_latent = self.visual_mlp(frame_features)
        
        '''
        # Fusion: 将机器人自身状态与提取出的周边图像隐特征拼接，送入多层感知机融合
        fusion_in = torch.cat((proprio, visual_latent), dim=-1)
        fusion_out = self.fusion_mlp(fusion_in) 
        
        # GRU Logic: 处理时间的序列特征，保存环境记忆
        batch_size = obs.shape[0]
        # 若是新环境启动或 batch/并行数量发生变化，重新初始化隐藏状态
        if not hasattr(self, 'actor_hidden_state') or self.actor_hidden_state.shape[1] != batch_size:
             # Init hidden state (1, batch, hidden)
             self.actor_hidden_state = torch.zeros(1, batch_size, self.gru_hidden_dim, device=obs.device)
        
        # Inputs: (batch, seq=1, input_size) GRU在PyTorch中期待3D的输入，即便序列长度=1
        gru_in = fusion_out.unsqueeze(1)
        # 前向GRu，保存输出并更新自身储存的隐特征
        gru_out, self.actor_hidden_state = self.gru(gru_in, self.actor_hidden_state)
        
        # Decoder
        # gru_out 维度还原: 去掉序列长度的壳 (batch, hidden)
        gru_out = gru_out.squeeze(1)
        # 将巨大的GRU状态降低维度，提纯为包含记忆的简洁向量，辅助主网判断
        decoded = self.gru_decoder(gru_out)
        '''
        
        # Concat for Actor MLP: 组合最新的单帧本体信息 + 10帧历史提取特征 + 深度图视觉特征
        return torch.cat((curr_proprio, history_latent, visual_latent), dim=-1)

    def _process_critic_obs(self, obs):
        # 处理 Critic 包含精准特权信息（地形扫描）的数据流
        if self.scan_dim <= 0:
            return obs
            
        proprio_priv = obs.narrow(1, 0, self.critic_proprio_priv_total_dim)
        scan = obs.narrow(1, self.critic_proprio_priv_total_dim, self.scan_total_dim)
        
        # 将带有历史栈的特权的地形扫描图经过高精度的简单全结合网络降维
        scan_latent = self.scan_encoder(scan) # [B, 32]
        # 和 Critic 特有的本体+全知特权信息拼接，作为 Value 评估的输入
        return torch.cat([proprio_priv, scan_latent], dim=1)

    def act(self, obs, **kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        features = self._process_actor_obs(obs)
        self.update_distribution(features)
        if self.distribution is None:
            raise RuntimeError("Distribution was not initialized after update.")
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):  #PPO更新函数update 会调用此函数
        if self.distribution is None:
            raise RuntimeError("Distribution was not initialized.")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs):    #play时调用
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        features = self._process_actor_obs(obs)
        return self.actor(features)

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        features = self._process_critic_obs(obs)
        return self.critic(features)