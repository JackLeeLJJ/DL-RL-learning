import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal, normal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
from flax import struct
import distrax
import gymnax
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import GymnaxWrapper
from brax import envs
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from evosax import OpenES, ParameterReshaper

import wandb

import sys
sys.path.insert(0, '..')
from purejaxrl.wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)
import time
import argparse
import pickle as pkl
import os

"""函数 wrap_brax_env 是一个用于应用一系列 Brax 环境包装器的函数。它接收一个环境 env 并根据需要将其包装为不同的功能模块。具体来说：

LogWrapper：这个包装器用于记录环境的日志，通常用于调试和分析。
ClipAction：该包装器对动作进行裁剪，确保它们在合法的范围内。
VecEnv：将环境包装为向量化环境，通常用于并行运行多个环境实例。
NormalizeVecObservation（如果 normalize_obs=True）：这个包装器用于标准化观察值，将其缩放到一定范围，通常用于加速训练过程。
NormalizeVecReward（如果 normalize_reward=True）：该包装器用于标准化奖励，帮助平衡不同任务的奖励尺度，通常也有助于稳定训练过程。
最后，函数返回经过所有包装器处理的环境 env。"""
def wrap_brax_env(env, normalize_obs=True, normalize_reward=True, gamma=0.99):
    """Apply standard set of Brax wrappers"""
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if normalize_obs:
        env = NormalizeVecObservation(env)
    if normalize_reward:
        env = NormalizeVecReward(env, gamma)
    return env

"""这个 BCAgentContinuous 类是一个用于连续动作空间的行为克隆（BC）智能体的模型。
它继承自 nn.Module，并通过定义前向传播方法 __call__ 来构建神经网络。"""
# Continuous action BC agent
class BCAgentContinuous(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    width: int = 64 #512 for Brax

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            self.width, kernel_init=normal(), bias_init=normal()
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.width, kernel_init=normal(), bias_init=normal()
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=normal(), bias_init=normal()
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        return pi

"""这个智能体的工作流程如下：

输入： 传入一个状态输入 x，它将通过一系列全连接层进行处理。
激活函数： 根据 activation 参数选择使用 ReLU 或 Tanh 作为激活函数。
网络结构：
第一个全连接层使用指定的宽度（默认为 64），并通过正态分布初始化权重和偏置。
后续两层全连接层依次处理每一层的输出，并分别应用激活函数。
最终输出是一个具有 action_dim 维度的全连接层，用来生成连续动作空间中的均值（actor_mean）。
标准差： 定义一个参数 log_std，其初始化为零，用来表示动作分布的标准差。通过 log_std 计算得到动作分布的标准差。
动作分布： 使用 distrax.MultivariateNormalDiag 来定义一个多元正态分布，均值是 actor_mean，标准差是 jnp.exp(actor_logtstd)（指数化处理以确保标准差为正值）。
最终，该模型返回一个表示连续动作空间分布的 pi，即一个多元正态分布对象，包含了根据当前输入计算得到的均值和标准差，用于生成动作。

此模型用于行为克隆任务，其中目标是学习一个策略，给定状态 x 时，生成一个动作分布以最大化执行的效果。"""


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
"""`Transition` 是一个使用 `NamedTuple` 定义的类，表示在强化学习中的一次状态转换（transition），它包含以下几个字段：
1. **done (jnp.ndarray):** 一个布尔数组，表示该状态转移是否已经结束（例如，任务是否完成，或是否达到了环境的终止状态）。
2. **action (jnp.ndarray):** 存储智能体在该状态下采取的动作。通常是一个数组，表示一个或多个动作的值。
3. **value (jnp.ndarray):** 存储在该状态下，智能体根据某个值函数（如价值函数）所估计的价值。这个值通常是智能体当前所处状态的一个数值表示，可能是 Q 值或者 V 值。
4. **reward (jnp.ndarray):** 存储智能体在该状态下获得的奖励。这个奖励通常是智能体与环境交互后得到的即时回报。
5. **log_prob (jnp.ndarray):** 存储智能体执行该动作时，动作选择概率的对数值。这个值用于强化学习中的策略梯度方法，以便计算梯度并更新策略。
6. **obs (jnp.ndarray):** 存储智能体在该状态下观察到的环境状态。通常是一个观测值，可以是图像、传感器数据或其他形式的状态描述。
7. **info (jnp.ndarray):** 存储与该状态转移相关的额外信息，这些信息可能用于调试、分析或其他目的。
这个 `Transition` 类型常用于强化学习的算法中，帮助存储和处理在每个时间步中智能体与环境之间的交互数据。通过这种结构，能够清晰地组织和访问相关的状态、动作、奖励等信息。"""

def make_train(config):
    """Create training function based on config. The returned function will:
    - Train a policy through BC
    - Evaluate the policy in the environment
    """
    config["NUM_UPDATES"] = config["UPDATE_EPOCHS"]

    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_OBS"]:
        env = NormalizeVecObservation(env)
    if config["NORMALIZE_REWARD"]:
        env = NormalizeVecReward(env, config["GAMMA"])

    # Do I need a schedule on the LR for BC?
    def linear_schedule(count):
        frac = 1.0 - (count // config["NUM_UPDATES"])
        return config["LR"] * frac

    def train(synth_data, action_labels, rng, nn_init_seed):
        """Train using BC on synthetic data with fixed action labels and evaluate on RL environment"""

        action_shape = env.action_space(env_params).shape[0]
        network = BCAgentContinuous(
            action_shape, activation=config["ACTIVATION"], width=config["WIDTH"]
        )

        # Special RNG with fixed seed used ONLY for the NN initialization (identical across rollouts & generations)
        nn_init_rng = jax.random.PRNGKey(nn_init_seed)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(nn_init_rng, init_x)

        assert (
                synth_data[0].shape == env.observation_space(env_params).shape
        ), f"Data of shape {synth_data[0].shape} does not match env observations of shape {env.observation_space(env_params).shape}"

        # Setup optimizer
        if config["ANNEAL_LR"]:
            lr = linear_schedule
        else:
            lr = config["LR"]
            
        if config["OPTIMIZER"] == "adam":
            optim = optax.adam(learning_rate=lr, eps=1e-5)
        else:
            optim = optax.sgd(learning_rate=lr)
            
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optim
        )

        # Train state carries everything needed for NN training
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # 2. BC TRAIN LOOP
        def _bc_train(train_state, rng):
            def _bc_update_step(bc_state, unused):
                train_state, rng = bc_state

                def _loss_and_acc(params, apply_fn, step_data, y_true, num_classes, grad_rng):
                    """Compute cross-entropy loss and accuracy.
                    y_true are prescribed actions, NOT action probabilities. Hence we take pi.log_prob(y_true)
                    """
                    pi = apply_fn(params, step_data)
                    y_pred = pi.sample(seed=grad_rng)

                    acc = jnp.mean(jnp.abs(y_pred - y_true))
                    log_prob = -pi.log_prob(y_true)
                    loss = jnp.sum(log_prob)
                    #                     loss = jnp.sum(jnp.abs(y_pred - y_true))
                    loss /= y_true.shape[0]

                    return loss, acc

                grad_fn = jax.value_and_grad(_loss_and_acc, has_aux=True)

                # Not needed if using entire dataset
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, len(action_labels))

                step_data = synth_data[perm]
                y_true = action_labels[perm]

                rng, state_noise_rng, act_noise_rng = jax.random.split(rng, 3)
                state_noise = jax.random.normal(state_noise_rng, step_data.shape)
                act_noise = jax.random.normal(act_noise_rng, y_true.shape)

                step_data = step_data + config["DATA_NOISE"] * state_noise
                y_true = y_true + config["DATA_NOISE"] * act_noise

                rng, grad_rng = jax.random.split(rng)

                loss_and_acc, grads = grad_fn(
                    train_state.params,
                    train_state.apply_fn,
                    step_data,
                    y_true,
                    action_shape,
                    grad_rng
                )
                train_state = train_state.apply_gradients(grads=grads)
                bc_state = (train_state, rng)
                return bc_state, loss_and_acc

            bc_state = (train_state, rng)
            bc_state, loss_and_acc = jax.lax.scan(
                _bc_update_step, bc_state, None, config["UPDATE_EPOCHS"]
            )
            loss, acc = loss_and_acc
            return bc_state, loss, acc

        rng, _rng = jax.random.split(rng)
        bc_state, bc_loss, bc_acc = _bc_train(train_state, _rng)
        train_state = bc_state[0]

        # Init envs
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        #         obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        obsv, env_state = env.reset(reset_rng, env_params)

        # 3. POLICY EVAL LOOP
        def _eval_ep(runner_state):
            # Environment stepper
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # Select Action
                rng, _rng = jax.random.split(rng)
                pi = train_state.apply_fn(train_state.params, last_obs)
                if config["GREEDY_ACT"]:
                    action = pi.argmax(
                        axis=-1
                    )  # if 2+ actions are equiprobable, returns first
                else:
                    action = pi.sample(seed=_rng)

                # Step env
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                #                 obsv, env_state, reward, done, info = jax.vmap(
                #                     env.step, in_axes=(0, 0, 0, None)
                #                 )(rng_step, env_state, action, env_params)
                obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, -1, reward, pi.log_prob(action), last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            metric = traj_batch.info
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = _eval_ep(runner_state)

        metric["bc_loss"] = bc_loss
        metric["bc_accuracy"] = bc_acc

        metric["states"] = synth_data
        metric["action_labels"] = action_labels
        metric["rng"] = rng

        return {"runner_state": runner_state, "metrics": metric}

    return train
"""这段代码定义了一个 `make_train` 函数，它根据传入的配置创建一个训练过程，主要进行以下任务：
1. **环境初始化：** 使用 `BraxGymnaxWrapper` 包装指定的环境，并根据配置选择是否对观测（`NormalizeVecObservation`）和奖励（`NormalizeVecReward`）进行归一化处理。同时，将 `ClipAction` 和 `LogWrapper` 应用于环境。
2. **学习率调度：** 根据配置是否启用学习率退火（`ANNEAL_LR`），如果启用，使用线性学习率调度器，随着训练步数的增加逐渐减少学习率。
3. **训练函数：** 在 `train` 函数中，使用 BC（行为克隆）方法训练一个连续动作的智能体。通过使用配置中指定的神经网络架构（`BCAgentContinuous`），并使用 Adam 或 SGD 优化器来更新网络参数。数据包括合成数据和相应的动作标签。训练过程中，还加入了一些噪声以增强训练的鲁棒性。
4. **策略评估：** 评估策略时，执行一定数量的环境步骤，并根据当前的策略生成动作并与环境交互。对于每个步骤，记录状态转换（`Transition`），并计算相关的度量（如损失和准确率）。
5. **返回结果：** 最后，返回训练的结果，包括训练状态和评估过程中得到的度量数据。
这个函数的核心思想是将 BC 训练与环境评估结合，通过模拟的合成数据训练一个智能体，并在真实环境中评估其性能。"""

def init_env(config):
    """Initialize environment"""
    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env = wrap_brax_env(env, normalize_obs=config["NORMALIZE_OBS"], normalize_reward=config["NORMALIZE_REWARD"])
    return env, env_params
"""
init_env 函数用于初始化并配置一个环境。它通过 BraxGymnaxWrapper 包装指定的环境，并使用 wrap_brax_env 函数进一步包装环境以支持观测和奖励的归一化。最后，函数返回配置好的环境和相应的环境参数。"""

def init_params(env, env_params, es_config):
    """Initialize dataset to be learned"""
    params = {
        "states": jnp.zeros((es_config["dataset_size"], *env.observation_space(env_params).shape)),
        "actions": jnp.zeros((es_config["dataset_size"], *env.action_space(env_params).shape))
    }
    param_reshaper = ParameterReshaper(params)
    return params, param_reshaper


def init_es(rng_init, param_reshaper, es_config):
    """Initialize OpenES strategy"""
    strategy = OpenES(
        popsize=es_config["popsize"],
        num_dims=param_reshaper.total_params,
        opt_name="adam",
        maximize=True,
    )
    # Replace state mean with real observations
    # state = state.replace(mean = sampled_data)

    es_params = strategy.default_params
    es_params = es_params.replace(sigma_init=es_config["sigma_init"], sigma_decay=es_config["sigma_decay"])
    state = strategy.initialize(rng_init, es_params)

    return strategy, es_params, state
"""
init_es 函数用于初始化一个 OpenES (Open-Ended Strategy) 策略。它通过给定的 es_config 配置（如种群大小、初始标准差和标准差衰减等），使用 Adam 优化器创建一个新的 OpenES 策略实例。然后，函数用 strategy.initialize 初始化策略状态，并返回策略、策略参数和初始化的状态。"""

def parse_arguments(argstring=None):
    """Parse arguments either from `argstring` if not None or from command line otherwise"""
    parser = argparse.ArgumentParser()
    # Default arguments should result in ~1600 return in Hopper

    # Outer loop args
    parser.add_argument(
        "--env",
        type=str,
        help="Brax environment name",
        default="hopper"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        help="Number of state-action pairs",
        default=4,
    )
    parser.add_argument(
        "--popsize",
        type=int,
        help="Number of state-action pairs",
        default=512
    )
    parser.add_argument(
        "--generations",
        type=int,
        help="Number of ES generations",
        default=200
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        help="Number of BC policies trained per candidate",
        default=1
    )
    parser.add_argument(
        "--sigma_init",
        type=float,
        help="Initial ES variance",
        default=0.03
    )
    parser.add_argument(
        "--sigma_decay",
        type=float,
        help="ES variance decay factor",
        default=1.0
    )

    # Inner loop args
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of BC epochs in the inner loop",
        default=20
    )
    parser.add_argument(
        "--eval_envs",
        type=int,
        help="Number of evaluation environments",
        default=16
    )
    parser.add_argument(
        "--activation",
        type=str,
        help="NN nonlinearlity type (relu/tanh)",
        default="tanh"
    )
    parser.add_argument(
        "--width",
        type=int,
        help="NN width",
        default=64
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="NN learning rate",
        default=5e-3
    )
    parser.add_argument(
        "--data_noise",
        type=float,
        help="Noise added to data during BC",
        default=0.0
    )
    parser.add_argument(
        "--normalize_obs",
        type=int,
        default=1
    )
    parser.add_argument(
        "--normalize_reward",
        type=int,
        default=1
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="adam or sgd",
        default="adam"
    )

    # Misc. args
    parser.add_argument(
        "--seed",
        type=int,
        help="RNG seed",
        default=1337
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        help="Num. generations between logs",
        default=1
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to save folder",
        default="../results/"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False
    )
    if argstring is not None:
        args = parser.parse_args(argstring.split())
    else:
        args = parser.parse_args()

    if args.folder[-1] != "/":
        args.folder = args.folder + "/"

    return args
"""parse_arguments 函数用于解析命令行参数或从给定的字符串中解析参数。它通过 argparse.ArgumentParser() 创建一个解析器，并定义了多种参数，如环境名称、数据集大小、种群大小、训练代数、学习率、神经网络结构等。函数首先检查是否传入了 argstring（如果传入，则从中解析参数，否则从命令行解析）。最终，返回解析后的参数 args，并确保文件路径 folder 以斜杠结尾。"""

def make_configs(args):
    config = {
        "LR": args.lr,  # 3e-4 for Brax?
        "NUM_ENVS": args.eval_envs,  # 8 # Num eval envs for each BC policy
        "NUM_STEPS": 1024,  # 128 # Max num eval steps per env
        "UPDATE_EPOCHS": args.epochs,  # Num BC gradient steps
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": args.activation,
        "WIDTH": args.width,
        "ENV_NAME": args.env,
        "ANNEAL_LR": False,  # False for Brax?
        "GREEDY_ACT": False,  # Whether to use greedy act in env or sample
        "DATA_NOISE": args.data_noise, # Add noise to data during BC training
        "ENV_PARAMS": {},
        "GAMMA": 0.99,
        "NORMALIZE_OBS": bool(args.normalize_obs),
        "NORMALIZE_REWARD": bool(args.normalize_reward),
        "DEBUG": args.debug,
        "SEED": args.seed,
        "FOLDER": args.folder,
        "OPTIMIZER": args.optimizer
    }
    es_config = {
        "popsize": args.popsize,  # Num of candidates (variations) generated every generation
        "dataset_size": args.dataset_size,  # Num of (s,a) pairs
        "rollouts_per_candidate": args.rollouts,  # 32 Num of BC policies trained per candidate
        "n_generations": args.generations,
        "log_interval": args.log_interval,
        "sigma_init": args.sigma_init,
        "sigma_decay": args.sigma_decay,
    }
    return config, es_config


def main(config, es_config):

    print("config")
    print("-----------------------------")
    for k, v in config.items():
        print(f"{k} : {v},")
    print("-----------------------------")
    print("ES_CONFIG")
    for k, v in es_config.items():
        print(f"{k} : {v},")

    # Setup wandb
    if not config["DEBUG"]:
        wandb_config = config.copy()
        wandb_config["es_config"] = es_config
        wandb_run = wandb.init(project="Dataset Neuroevolution", config=wandb_config)
        wandb.define_metric("D")
        wandb.summary["D"] = es_config["dataset_size"]
        #     wandb.define_metric("mean_fitness", summary="last")
        #     wandb.define_metric("max_fitness", summary="last")

    # Init environment and dataset (params)
    env, env_params = init_env(config)
    params, param_reshaper = init_params(env, env_params, es_config)

    rng = jax.random.PRNGKey(config["SEED"])

    # Initialize OpenES Strategy
    rng, rng_init = jax.random.split(rng)
    strategy, es_params, state = init_es(rng_init, param_reshaper, es_config)

    # Set up vectorized fitness function
    train_fn = make_train(config)

    def single_seed_BC(rng_input, dataset, action_labels):
        # Train using a fixed seed for initializing NNs across all of training
        out = train_fn(dataset, action_labels, rng_input, nn_init_seed=config["SEED"]+1337)
        return out

    multi_seed_BC = jax.vmap(single_seed_BC, in_axes=(0, None, None))  # Vectorize over seeds
    train_and_eval = jax.jit(
        jax.vmap(multi_seed_BC, in_axes=(None, 0, 0)))  # Vectorize over datasets

    if len(jax.devices()) > 1:
        # If available, distribute over multiple GPUs
        train_and_eval = jax.pmap(train_and_eval, in_axes=(None, 0, 0))

    start = time.time()
    lap_start = start
    fitness_over_gen = []
    max_fitness_over_gen = []
    for gen in range(es_config["n_generations"]):
        # Gen new dataset
        rng, rng_ask, rng_inner = jax.random.split(rng, 3)
        datasets, state = jax.jit(strategy.ask)(rng_ask, state, es_params)
        # Eval fitness
        batch_rng = jax.random.split(rng_inner, es_config["rollouts_per_candidate"])
        # Preemptively overwrite to reduce memory load
        out = None
        returns = None
        dones = None
        fitness = None
        shaped_datasets = None

        with jax.disable_jit(config["DEBUG"]):
            shaped_datasets = param_reshaper.reshape(datasets)

            out = train_and_eval(batch_rng, shaped_datasets["states"], shaped_datasets["actions"])

            returns = out["metrics"]["returned_episode_returns"]  # dim=(popsize, rollouts, num_steps, num_envs)
            ep_lengths = out["metrics"]["returned_episode_lengths"]
            dones = out["metrics"]["returned_episode"]  # same dim, True for last steps, False otherwise

            mean_ep_length = (ep_lengths * dones).sum(axis=(-1, -2, -3)) / dones.sum(
                axis=(-1, -2, -3))
            mean_ep_length = mean_ep_length.flatten()

            # Division by zero, watch out
            fitness = (returns * dones).sum(axis=(-1, -2, -3)) / dones.sum(
                axis=(-1, -2, -3))  # fitness, dim = (popsize)
            fitness = fitness.flatten()  # Necessary if pmap-ing to 2+ devices
        #         fitness = jnp.minimum(fitness, fitness.mean()+40)

        # Update ES strategy with fitness info
        state = jax.jit(strategy.tell)(datasets, fitness, state, es_params)
        fitness_over_gen.append(fitness.mean())
        max_fitness_over_gen.append(fitness.max())

        # Logging
        if gen % es_config["log_interval"] == 0 or gen == 0:
            lap_end = time.time()
            if len(jax.devices()) > 1:
                bc_loss = out["metrics"]["bc_loss"][:, :, :, -1]
                bc_acc = out["metrics"]["bc_accuracy"][:, :, :, -1]
            else:
                bc_loss = out["metrics"]["bc_loss"][:, :, -1]
                bc_acc = out["metrics"]["bc_accuracy"][:, :, -1]

            print(
                f"Gen: {gen}, Fitness: {fitness.mean():.2f} +/- {fitness.std():.2f}, "
                + f"Best: {state.best_fitness:.2f}, BC loss: {bc_loss.mean():.2f} +/- {bc_loss.std():.2f}, "
                + f"BC mean error: {bc_acc.mean():.2f} +/- {bc_acc.std():.2f}, Lap time: {lap_end - lap_start:.1f}s"
            )
            if not config["DEBUG"]:
                wandb.log({
                    f"{config['ENV_NAME']}:mean_fitness": fitness.mean(),
                    f"{config['ENV_NAME']}:fitness_std": fitness.std(),
                    f"{config['ENV_NAME']}:max_fitness": fitness.max(),
                    "mean_ep_length": mean_ep_length.mean(),
                    "max_ep_length": mean_ep_length.max(),
                    "mean_fitness": fitness.mean(),
                    "max_fitness": fitness.max(),
                    "BC_loss": bc_loss.mean(),
                    "BC_accuracy": bc_acc.mean(),
                    "Gen time": lap_end - lap_start,
                })
            lap_start = lap_end
    print(f"Total time: {(lap_end - start) / 60:.1f}min")

    data = {
        "state": state,
        "fitness_over_gen": fitness_over_gen,
        "max_fitness_over_gen": max_fitness_over_gen,
        "fitness": fitness,
        "config": config,
        "es_config": es_config
    }

    directory = config["FOLDER"]
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = directory + "data.pkl"
    file = open(filename, 'wb')
    pkl.dump(data, file)
    file.close()


def train_from_arg_string(argstring):
    """Launches training from an argument string of the form
    `--env humanoid --popsize 1024 --epochs 200 ...`
    Main use case is in conjunction with Submitit for creating job arrays
    """
    args = parse_arguments(argstring)
    config, es_config = make_configs(args)
    main(config, es_config)


if __name__ == "__main__":
    args = parse_arguments()
    config, es_config = make_configs(args)
    main(config, es_config)

