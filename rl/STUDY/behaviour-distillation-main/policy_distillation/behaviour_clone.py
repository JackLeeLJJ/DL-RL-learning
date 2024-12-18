import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import time

class BCAgent(nn.Module):
    """Network architecture. Matches MinAtar PPO agent from PureJaxRL"""

    action_dim: Sequence[int] #有几个动作维度
    activation: str = "tanh" #激活函数
    width: int = 64  #神经网络神经元数目

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            self.width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x) #x是输入张量（上一层网络输出），kernel_init是初始化权重矩阵，bias是偏置。dense用于构建线性层
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        return actor_mean


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    """Create training function based on config."""
    config["NUM_UPDATES"] = config["UPDATE_EPOCHS"]

    env, env_params = gymnax.make(config["ENV_NAME"])
    env_params = env_params.replace(**config["ENV_PARAMS"])#**是解包符号，把config中的内容传到replace中
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    # Do I need a schedule on the LR for BC?
    def linear_schedule(count):
        frac = 1.0 - (count // config["NUM_UPDATES"])#count表示当前的更新次数或训练轮数。它用于计算衰减比例。NUM_UPDATES表示总的更新次数（例如，整个训练过程的最大更新次数）
        return config["LR"] * frac


    def train(synth_data, action_labels, rng):
        """Train using BC on synthetic data with fixed action labels and evaluate on RL environment
        在合成数据上使用行为克隆（BC）训练，使用固定的动作标签，并在强化学习（RL）环境中进行评估。
        """

        # 1. INIT NETWORK AND TRAIN STATE
        # TODO: This is hacky. Fix it to handle continuous actions elegantly

        #判断动作空间连续还是离散
        if "Continuous" in config["ENV_NAME"] or "Brax" in config["ENV_NAME"]:
            action_shape = env.action_space().shape[0]
            is_continuous = True
        else:
            action_shape = env.action_space().n
            is_continuous = False
        network = BCAgent(
            action_shape, activation=config["ACTIVATION"], width=config["WIDTH"]
        )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        """
        init_x：一个示例输入，通常用于确定网络输入的形状，网络根据这个输入的形状来初始化相应的参数。这里，init_x 是一个全零数组，其形状与环境的观测空间一致。
        network.init 会返回网络的初始化参数，通常这些参数包括网络的权重和偏置。
        network_params 是一个包含网络初始化参数的对象，可以进一步用于训练或评估神经网络。"""

        assert (
                synth_data[0].shape == env.observation_space(env_params).shape
        ), f"Data of shape {synth_data[0].shape} does not match env observations of shape {env.observation_space(env_params).shape}"

        # Setup optimizer
        if config["ANNEAL_LR"]:#使用权重衰减
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),#梯度剪裁
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        # Train state carries everything needed for NN training
        train_state = TrainState.create(
            apply_fn=network.apply,#网络的前向传播函数，通常用于将输入数据传递到网络中并进行计算，得到输出
            params=network_params,
            tx=tx,
        )

        # 2. BC TRAIN LOOP
        def _bc_train(train_state, rng):
            def _bc_update_step(bc_state, unused):
                train_state, rng = bc_state

                def _loss_and_acc(params, apply_fn, step_data, y_true, num_classes):
                    """Compute cross-entropy loss and accuracy."""
                    y_pred = apply_fn(params, step_data)
                    acc = jnp.mean(jnp.argmax(y_pred, axis=-1) == y_true)
                    labels = jax.nn.one_hot(y_true, num_classes)
                    loss = -jnp.sum(labels * jax.nn.log_softmax(y_pred))#计算每个样本类别的交叉熵损失
                    loss /= labels.shape[0]#每个样本的平均损失
                    return loss, acc

                # TODO: Add loss function for continuous case --> XENT?
                grad_fn = jax.value_and_grad(_loss_and_acc, has_aux=True) #value_and_grad 同时计算某个函数的输出值（函数值）和该函数关于输入的梯度（导数）
                #has_aux=True：表示除了返回梯度之外，value_and_grad 还会返回函数的 辅助输出。如果函数返回多个值或一些额外的中间变量，设置 has_aux=True 可以同时返回这些值。
                # 调用 grad_fn 时：loss, grad, acc = grad_fn(params, apply_fn, step_data, y_true, num_classes)



                # Not needed if using entire dataset
                rng, perm_rng = jax.random.split(rng) #分裂随机数生成器rng：原来的随机数生成器（可以继续用于其他任务）。perm_rng：新分裂出的随机数生成器，用于生成随机排列的索引。
                perm = jax.random.permutation(perm_rng, len(action_labels))#生成一个随机排列的索引
                step_data = synth_data[perm]
                y_true = action_labels[perm]

                loss_and_acc, grads = grad_fn(
                    train_state.params,
                    train_state.apply_fn,
                    step_data,
                    y_true,
                    action_shape,
                )
                train_state = train_state.apply_gradients(grads=grads)#利用 计算得到的梯度 更新当前 训练状态 中的 模型参数。通过应用梯度更新，模型的参数会逐步朝着减少损失的方向调整，从而提高模型的性能。
                bc_state = (train_state, rng)
                return bc_state, loss_and_acc

            bc_state = (train_state, rng)
            bc_state, loss_and_acc = jax.lax.scan(
                _bc_update_step, bc_state, None, config["UPDATE_EPOCHS"]
            )#jax.lax.scan 将反复调用 _bc_update_step 函数，执行 UPDATE_EPOCHS 次
            loss, acc = loss_and_acc
            return bc_state, loss, acc

        rng, _rng = jax.random.split(rng)
        bc_state, bc_loss, bc_acc = _bc_train(train_state, _rng)
        train_state = bc_state[0]

        # Init envs
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])#将 _rng 分裂成多个子随机数生成器，数量为 config["NUM_ENVS"]，即环境的数量
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)#并行地初始化多个环境，并为每个环境生成不同的初始状态。

        # 3. POLICY EVAL LOOP
        def _eval_ep(runner_state):#执行一个 回合（episode） 的评估步骤
            # Environment stepper
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # Select Action
                rng, _rng = jax.random.split(rng)
                pi = train_state.apply_fn(train_state.params, last_obs)#last_obs：表示当前步骤的观测，即在当前时刻代理所看到的环境状态。
                if config["GREEDY_ACT"]:
                    action = pi.argmax(
                        axis=-1
                    )  # if 2+ actions are equiprobable, returns first  返回概率最大的动作
                else:
                    probs = distrax.Categorical(logits=pi)
                    action = probs.sample(seed=_rng)#随机采样

                # Step env
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)#
                )(rng_step, env_state, action, env_params)#使用 JAX 的 vmap 来并行化多个环境的 env.step 操作，并同时更新多个环境的状态、奖励等信息。
                transition = Transition(
                    done, action, -1, reward, jax.nn.log_softmax(pi), last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(#traj_batch：记录环境在每个时间步的变化（轨迹批次），通常用于分析或计算策略的表现。
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            metric = traj_batch.info
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = _eval_ep(runner_state)

        metric["bc_loss"] = bc_loss
        metric["bc_accuracy"] = bc_acc #返回行为克隆最终的损失和准确率

        return {"runner_state": runner_state, "metrics": metric}

    return train


