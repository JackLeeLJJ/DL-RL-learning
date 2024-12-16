import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces
from brax import envs


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info
"""`FlattenObservationWrapper` 是一个用于扁平化环境观察空间的包装器。它继承自 `GymnaxWrapper`，并重写了环境的 `observation_space`、`reset` 和 `step` 方法。

- **`observation_space`**: 该方法检查环境的观察空间是否为 `Box` 类型，并返回一个形状为一维的 `Box` 空间，维度是原始观察空间的元素数量（即将原始空间展平）。
  
- **`reset`**: 该方法调用环境的 `reset` 方法获取初始观察和状态，并将观察展平为一维数组返回。

- **`step`**: 该方法调用环境的 `step` 方法执行一步操作，返回展平后的观察值，并包含环境的状态、奖励、是否完成和额外信息。

通过这种方式，环境的观察空间被扁平化为一维数组，方便在需要处理一维输入的场景（如神经网络输入）中使用。"""


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int

"""LogWrapper 的作用是在与环境交互时，记录每个回合的累计奖励、回合长度和时间步数，并在回合结束时保存这些信息以供后续分析。它通过修改 reset 和 step 方法来实现这一目标，并将相关数据保存在 info 字典中。"""
"""reset 函数:

功能: 重置环境并初始化回合日志。
操作: 调用环境的 reset 方法获取初始观察和环境状态，然后创建一个 LogEnvState 对象来存储回合的累计奖励、回合长度等信息。
返回: 返回初始观察和 LogEnvState（包含日志信息）。
step 函数:

功能: 执行一步环境交互，并更新回合日志。
操作: 调用环境的 step 方法获取新的观察、奖励、完成状态等信息，然后更新回合的累计奖励和长度。如果回合结束，保存本回合的最终奖励和长度。
返回: 返回新的观察、更新后的环境状态、奖励、完成状态和附加信息（包括日志数据）。"""
class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info

"""### 总体总结：
`BraxGymnaxWrapper` 类将 Brax 环境适配为 Gymnax 环境，提供了标准化的接口，如 `reset`、`step`，并通过 `observation_space` 和 `action_space` 函数定义了环境的状态和动作空间。

### 各个函数总结：
1. **`__init__`**: 初始化 Brax 环境并设置环境的动作和观察空间。
2. **`reset`**: 重置环境并返回初始的观察和状态。
3. **`step`**: 执行一个环境步骤并返回新的观察、状态、奖励、完成状态和附加信息。
4. **`observation_space`**: 返回环境的观察空间，表示为一个 `spaces.Box`。
5. **`action_space`**: 返回环境的动作空间，表示为一个 `spaces.Box`。"""
class BraxGymnaxWrapper:
    def __init__(self, env_name, backend="positional"):
        env = envs.get_environment(env_name=env_name, backend=backend)
        env = envs.wrappers.training.EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = envs.wrappers.training.AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )

"""总体总结：
ClipAction 类是对 Gymnax 环境的包装器，主要功能是将传入的动作限制在指定的区间范围内，以确保动作不会超出预定义的上下限。

各个函数总结：
__init__: 初始化包装器，设置动作的最小值 (low) 和最大值 (high)。
step: 在执行环境步骤前，将动作进行裁剪，确保动作在 [low, high] 范围内，然后调用环境的 step 函数。"""
class ClipAction(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)

"""### 总体总结：
`TransformObservation` 类是对 Gymnax 环境的包装器，主要功能是在每次环境重置和每个步骤中应用自定义的观察转换函数，对环境返回的观察数据进行变换。

### 各个函数总结：
1. **`__init__`**: 初始化包装器，接收一个自定义的观察转换函数 `transform_obs`。
2. **`reset`**: 重置环境并应用 `transform_obs` 函数变换初始观察数据。
3. **`step`**: 执行环境步骤，并对返回的观察数据应用 `transform_obs` 函数进行变换。"""
class TransformObservation(GymnaxWrapper):
    def __init__(self, env, transform_obs):
        super().__init__(env)
        self.transform_obs = transform_obs

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self.transform_obs(obs), state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self.transform_obs(obs), state, reward, done, info

"""### 总体总结：
`TransformReward` 类是对 Gymnax 环境的包装器，主要功能是在每次环境步骤中应用自定义的奖励转换函数，对环境返回的奖励数据进行变换。

### 各个函数总结：
1. **`__init__`**: 初始化包装器，接收一个自定义的奖励转换函数 `transform_reward`。
2. **`step`**: 执行环境步骤，并对返回的奖励数据应用 `transform_reward` 函数进行变换。"""
class TransformReward(GymnaxWrapper):
    def __init__(self, env, transform_reward):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return obs, state, self.transform_reward(reward), done, info

"""### 总体总结：
`VecEnv` 类是对 Gymnax 环境的包装器，允许在多个环境实例中并行执行操作。它通过 `jax.vmap` 将环境的 `reset` 和 `step` 操作向量化，支持批量处理。

### 各个函数总结：
1. **`__init__`**: 初始化包装器，通过 `jax.vmap` 向量化 `reset` 和 `step` 方法，实现多个环境实例的并行操作。"""
class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

"""### 总体总结：
`NormalizeVecObsEnvState` 是一个数据类，用于存储与环境观察值归一化相关的状态信息，包括均值、方差、样本计数以及环境的内部状态。

### 各个字段总结：
1. **`mean`**: 观察值的均值，用于归一化。
2. **`var`**: 观察值的方差，用于归一化。
3. **`count`**: 样本计数，记录累计的观察样本数量。
4. **`env_state`**: 环境的内部状态，保存环境当前的状态信息。"""
@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: environment.EnvState

"""### 总体总结：
`NormalizeVecObservation` 类是对环境观察值进行归一化处理的包装器。它在每个时间步通过更新均值、方差以及样本计数来归一化观察值，从而确保环境的输入数据在训练过程中具有一致的尺度。

### 各个函数总结：
1. **`reset`**: 在重置环境时，计算并更新归一化的均值、方差和样本计数，并将这些信息保存在 `NormalizeVecObsEnvState` 中。最后返回归一化后的观察值和新的环境状态。
   
2. **`step`**: 在每个时间步中，计算新的观察值的均值、方差和样本计数，基于这些数据更新归一化信息。返回归一化后的观察值、更新后的环境状态、奖励、结束标志以及额外信息。"""
class NormalizeVecObservation(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        state = NormalizeVecObsEnvState(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-8),
            state,
            reward,
            done,
            info,
        )

"""`NormalizeVecRewEnvState` 是一个数据类，用于存储与环境奖励归一化相关的状态信息。它包含以下字段：

1. **`mean`**: 奖励的均值，用于奖励归一化的计算。
2. **`var`**: 奖励的方差，帮助在计算归一化时使用。
3. **`count`**: 样本数量，用于更新均值和方差时的加权。
4. **`return_val`**: 累积的返回值，用于跟踪当前回合的总奖励。
5. **`env_state`**: 环境的状态，用于保持环境的其他状态信息。

这个数据类被用于奖励归一化过程中的状态跟踪，确保奖励的均值和方差随着时间的推移而更新。"""
@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState

"""
函数总结：
__init__(self, env, gamma): 初始化NormalizeVecReward类，设置环境和折扣因子gamma。

reset(self, key, params=None): 重置环境，初始化状态，包括均值、方差、计数和回报值。

step(self, key, state, action, params=None): 执行环境的一步，计算折扣回报，并根据当前回报值更新均值和方差，用于标准化奖励。

总体总结：
NormalizeVecReward类是一个包装器，用于在Gymnax环境中标准化奖励。它通过维护回报的均值和方差来稳定训练过程，使得每一步的奖励更为一致，从而帮助强化学习算法更好地学习。"""
class NormalizeVecReward(GymnaxWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info
