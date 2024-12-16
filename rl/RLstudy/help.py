# import gym
# env = gym.make('Pendulum-v0'
#                )
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()


# #查看gym中有哪些环境
# from gym import envs
# env_specs = envs.registry.all()
# env_ids = [env_spec.id for env_spec in env_specs]
# print(env_ids)
# env_name='Ant-v2'
# if env_name in env_ids:
#     print(env_name)
# else:
#     print('not exist error')

#
# import gym
# import matplotlib.pyplot as plt
# from matplotlib import animation
#
# # 创建环境
# env = gym.make('Ant-v2')
#
# # 用来存储每一帧图像
# frames = []
#
# # 采样一个 episode 的数据
# for i_episode in range(1):  # 只记录一个 episode
#     observation = env.reset()
#     for t in range(100):  # 最多记录 100 步
#         frame = env.render(mode="rgb_array")  # 渲染为图像数据
#         frames.append(frame)  # 保存当前帧
#         action = env.action_space.sample()  # 随机选择动作
#         observation, reward, done, info = env.step(action)
#         if done:
#             print(f"Episode finished after {t+1} timesteps")
#             break
#
# env.close()
#
# # 创建动画
# fig = plt.figure(figsize=(8, 6))
# plt.axis("off")  # 不显示坐标轴
#
# # 显示第一帧
# img = plt.imshow(frames[0])
#
# # 更新函数
# def update_frame(i):
#     img.set_data(frames[i])
#     return img,
#
# # 创建动画
# ani = animation.FuncAnimation(fig, update_frame, frames=len(frames), interval=50, blit=True)
#
# # 保存为 GIF 或显示
# ani.save("ant_v2_animation.gif", writer="pillow")
# plt.show()


#输出环境动作空间和状态空间的形状
import gym
env = gym.make('BipedalWalker-v2')
print(env.action_space)
print(env.observation_space)



# import gym
# from gym.spaces import Discrete, Box
#
#
# def check_environment_type(env):
#     """
#     判断强化学习环境是离散的还是连续的。
#
#     :param env: 强化学习环境（gym.Env）
#     :return: 环境类型的字符串，'discrete' 或 'continuous'
#     """
#     # 检查动作空间类型
#     if isinstance(env.action_space, Discrete):
#         action_type = 'discrete'
#     elif isinstance(env.action_space, Box):
#         action_type = 'continuous'
#     else:
#         action_type = 'unknown'
#
#     # 检查观察空间类型
#     if isinstance(env.observation_space, Discrete):
#         observation_type = 'discrete'
#     elif isinstance(env.observation_space, Box):
#         observation_type = 'continuous'
#     else:
#         observation_type = 'unknown'
#
#     return action_type, observation_type
#
#
# # 示例用法
# if __name__ == "__main__":
#     # 选择一个环境（你可以替换成任何其他的Gym环境）
#     env = gym.make('CartPole-v1')
#
#     action_type, observation_type = check_environment_type(env)
#
#     print(f"Action space is {action_type}, Observation space is {observation_type}")


