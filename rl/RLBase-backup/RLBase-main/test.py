import gymnasium as gym
from ale_py import ALEInterface

# 创建环境
env = gym.make("SpaceInvaders-v4", render_mode='human')  # 使用最新版环境 ID 格式

# 初始化环境
env.reset()

# 主循环：随机动作
try:
    while True:
        # 随机选择一个动作
        action = env.action_space.sample()
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        # 如果游戏结束（无论是目标达成还是超时），重置环境
        if terminated or truncated:
            env.reset()
except KeyboardInterrupt:
    print("用户手动停止程序")

# 关闭环境
env.close()