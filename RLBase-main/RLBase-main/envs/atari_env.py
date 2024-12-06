# 导入必要的库
import gymnasium as gym  # 用于创建和管理强化学习环境
from ale_py import ALEInterface  # Atari Learning Environment (ALE) 的接口库


# 定义一个函数，用于创建 Atari 游戏环境
def create_atari_env(env_id: str) -> gym.Env:
    # ALEInterface 是 Atari 游戏的底层接口，提供了对 Atari Learning Environment 的访问
    ale = ALEInterface()

    # 注册 ALE 支持的 Atari 游戏环境到 Gym,使其可以通过 Gym 的 API 使用。
    gym.register_envs(ale)

    # 使用 Gym 提供的 make 方法创建指定的 Atari 环境
    env = gym.make(env_id)

    # 返回创建好的环境实例
    return env
