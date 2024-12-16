import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from common.buffers import RolloutBuffer
from common.on_policy_algorithm import OnPolicyAlgorithm
from common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from common.type_aliases import GymEnv, MaybeCallback, Schedule
from common.utils import explained_variance, get_schedule_fn

SelfPPO = TypeVar("SelfPPO", bound="PPO")
'''以下是对这些引入的包和模块的作用的推测和解释：

### 1. **`warnings`**
- **作用**：用于处理警告信息，比如在代码中提示用户某些操作可能导致问题。通常用来提醒用户使用了过时的功能或者配置错误。

### 2. **`typing`**
- **模块说明**：提供类型提示功能，用于提高代码的可读性和维护性。
  - **`Any`**：表示任意类型。
  - **`ClassVar`**：用于声明类变量，而非实例变量。
  - **`Optional`**：表示类型可以是指定的类型，也可以是 `None`。
  - **`TypeVar`**：定义泛型类型，用于类或函数的泛型设计。
  - **`Union`**：表示变量可能是多种类型之一。

### 3. **`numpy as np`**
- **作用**：提供数组操作、高效数值计算、统计分析等功能。在强化学习中，常用于处理观测值、动作、奖励和状态等数据。

### 4. **`torch as th`**
- **作用**：`PyTorch` 是一个流行的深度学习框架。
  - 提供张量操作（类似 NumPy，但支持 GPU 加速）。
  - 支持构建神经网络，用于强化学习中的策略网络和值函数网络。

### 5. **`gymnasium.spaces`**
- **模块说明**：`gymnasium` 是 OpenAI Gym 的新版本。
  - **`spaces`**：定义动作空间和状态空间的类型，用于描述强化学习环境中的输入输出范围。
    - **`Box`**：连续动作空间。
    - **`Discrete`**：离散动作空间。
    - **`MultiDiscrete`**：多维离散动作空间。
    - **`MultiBinary`**：多维二元动作空间。

### 6. **`torch.nn.functional as F`**
- **作用**：提供一系列用于神经网络的函数，比如激活函数（`relu`）、损失函数（`mse_loss`）、卷积操作（`conv2d`）等。在强化学习中，这些功能用于构建和训练策略网络。

### 7. **`common.buffers`**
- **模块说明**：用户自定义模块，可能实现了经验缓存的功能。
  - **`RolloutBuffer`**：通常用于存储交互的状态、动作、奖励等，用于策略的训练更新。

### 8. **`common.on_policy_algorithm`**
- **模块说明**：用户自定义模块，可能包含与“基于策略的算法”（on-policy）相关的实现。
  - 比如 PPO、A2C 等强化学习算法的核心逻辑。

### 9. **`common.policies`**
- **模块说明**：用户自定义模块，可能包含不同策略（Policy）的实现。
  - **`ActorCriticCnnPolicy`**：基于卷积神经网络的 Actor-Critic 策略。
  - **`ActorCriticPolicy`**：标准的 Actor-Critic 策略。
  - **`BasePolicy`**：可能是所有策略的基类。
  - **`MultiInputActorCriticPolicy`**：支持多输入的 Actor-Critic 策略（如图像和其他特征的联合输入）。

### 10. **`common.type_aliases`**
- **模块说明**：用户自定义模块，定义了一些常用的类型别名。
  - **`GymEnv`**：可能是环境的类型别名。
  - **`MaybeCallback`**：可能是用于定义回调函数类型的别名。
  - **`Schedule`**：可能是学习率或其他调度策略的类型别名。

### 11. **`common.utils`**
- **模块说明**：用户自定义模块，可能包含辅助工具函数。
  - **`explained_variance`**：计算解释方差，用于评估值函数的表现。
  - **`get_schedule_fn`**：将常量或函数转换为调度函数，用于动态调整超参数。

### 12. **`SelfPPO = TypeVar("SelfPPO", bound="PPO")`**
- **作用**：定义了一个泛型 `SelfPPO`，用于表示当前类 `PPO` 的子类实例。这种定义常用于面向对象编程中的方法链设计。

---

### 小结
这些引入的模块和包主要分为以下几类：
1. **标准库**：如 `warnings` 和 `typing`，提供基本的工具和类型支持。
2. **第三方库**：如 `numpy`、`torch` 和 `gymnasium`，用于数值计算、深度学习和强化学习环境交互。
3. **用户自定义模块**：如 `common.*`，包含算法实现、策略定义、工具函数等，用于具体的强化学习任务。

这些模块的结合，形成了一个强化学习算法实现的框架，涵盖了从环境交互到策略优化的全流程。'''

class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    '''
    def __init__(
            policy: Union[str, type[ActorCriticPolicy]],  # 策略类型，可以是字符串（如“MlpPolicy”）或自定义策略类
            env: Union[GymEnv, str],  # 强化学习环境实例或环境名称字符串
            learning_rate: Union[float, Schedule] = 3e-4,  # 学习率，可以是固定值或学习率调度函数
            n_steps: int = 2048,  # 每次更新使用的环境交互步数
            batch_size: int = 64,  # 每次优化时使用的批量大小
            n_epochs: int = 10,  # 每次更新时进行的优化迭代次数
            gamma: float = 0.99,  # 折扣因子，决定未来奖励的权重
            gae_lambda: float = 0.95,  # GAE（广义优势估计）的 lambda 参数，平衡偏差与方差
            clip_range: Union[float, Schedule] = 0.2,  # PPO 算法中的裁剪范围，控制策略更新幅度
            clip_range_vf: Union[None, float, Schedule] = None,  # 值函数的裁剪范围（默认不裁剪）
            normalize_advantage: bool = True,  # 是否对优势函数进行归一化
            ent_coef: float = 0.0,  # 策略熵的权重，用于鼓励策略的探索性
            vf_coef: float = 0.5,  # 值函数损失的权重
            max_grad_norm: float = 0.5,  # 梯度的最大范数，用于梯度裁剪
            use_sde: bool = False,  # 是否使用状态相关的探索（State-Dependent Exploration）
            sde_sample_freq: int = -1,  # 状态相关探索的采样频率（-1 表示每次更新都重新采样）
            rollout_buffer_class: Optional[type[RolloutBuffer]] = None,  # 自定义 rollout buffer 的类
            rollout_buffer_kwargs: Optional[dict[str, Any]] = None,  # rollout buffer 的额外参数
            target_kl: Optional[float] = None,  # 目标 KL 散度值，用于早停（默认关闭）
            stats_window_size: int = 100,  # 用于统计回报和损失的窗口大小
            tensorboard_log: Optional[str] = None,  # TensorBoard 日志文件存储路径
            policy_kwargs: Optional[dict[str, Any]] = None,  # 策略的额外参数
            verbose: int = 0,  # 日志输出的详细程度（0: 无输出, 1: 信息, 2: 调试）
            seed: Optional[int] = None,  # 随机种子，用于实验复现
            device: Union[th.device, str] = "auto",  # 使用的设备（“cpu”、“cuda” 或 “auto”）
            _init_setup_model: bool = True,  # 是否在初始化时立即设置模型（内部参数）
    ):
        # 父类初始化，设置共享参数
        super().__init__(
            policy,  # 策略类型
            env,  # 环境
            learning_rate=learning_rate,  # 学习率
            n_steps=n_steps,  # 每次更新的步数
            gamma=gamma,  # 折扣因子
            gae_lambda=gae_lambda,  # GAE 的 lambda 参数
            ent_coef=ent_coef,  # 策略熵权重
            vf_coef=vf_coef,  # 值函数损失权重
            max_grad_norm=max_grad_norm,  # 梯度裁剪
            use_sde=use_sde,  # 是否使用状态相关探索
            sde_sample_freq=sde_sample_freq,  # 状态相关探索采样频率
            rollout_buffer_class=rollout_buffer_class,  # rollout buffer 类
            rollout_buffer_kwargs=rollout_buffer_kwargs,  # rollout buffer 参数
            stats_window_size=stats_window_size,  # 回报统计窗口大小
            tensorboard_log=tensorboard_log,  # TensorBoard 日志路径
            policy_kwargs=policy_kwargs,  # 策略参数
            verbose=verbose,  # 日志输出详细程度
            device=device,  # 运行设备
            seed=seed,  # 随机种子
            _init_setup_model=False,  # 是否立即初始化模型
            supported_action_spaces=(  # 支持的动作空间类型
                spaces.Box,  # 连续动作空间
                spaces.Discrete,  # 离散动作空间
                spaces.MultiDiscrete,  # 多维离散动作空间
                spaces.MultiBinary,  # 多维二元动作空间
            ),
        )
'''
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
