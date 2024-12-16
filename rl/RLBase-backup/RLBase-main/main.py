import os  # 用于文件和路径操作
import numpy as np  # 数值计算库，主要用于数组操作
import random  # Python 自带的随机数生成模块
import time  # 用于测量运行时间
import wandb  # 用于实验追踪和可视化的库
import argparse  # 用于解析命令行参数
import torch  # PyTorch 深度学习框架
from envs.atari_env import create_atari_env  # 导入自定义的 Atari 环境创建函数
from algo import PPO  # 导入 PPO 算法实现

# 设置随机种子以保证实验的可复现性
def set_seed(seed: int):
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
    random.seed(seed)  # 设置 Python 内置随机数生成器的种子
    np.random.seed(seed)  # 设置 NumPy 的随机数生成器的种子

# 初始化 WandB 进行实验追踪
def initialize_wandb(args):
    return wandb.init(
        project="gen",  # 项目名称，用户需要自行定义
        name=f"{args.env}_{args.policy_type}_{args.seed}",  # 实验运行的名称
        config=vars(args),  # 自动将命令行参数作为配置记录到 WandB
        sync_tensorboard=True,  # 同步 TensorBoard 日志
        save_code=True,  # 保存当前代码到 WandB
        notes="Training PPO on Atari environment",  # 添加一些可选的实验说明
        mode='online'  # 设置 WandB 为在线模式
    )

# 主程序，负责训练
def main(args):
    # 从命令行参数中获取设置
    env_id = args.env  # 获取环境名称
    policy_type = args.policy_type  # 获取 PPO 的策略类型
    total_timesteps = args.total_timesteps  # 总训练时间步数
    use_cuda = args.use_cuda  # 是否使用 GPU 加速

    # 如果用户没有指定随机种子，则随机生成一个
    if args.seed is None:
        args.seed = np.random.randint(0, 10000)  # 生成一个 0 到 10000 的随机整数
    set_seed(args.seed)  # 设置随机种子

    # 根据设备可用性选择运行设备（CUDA 或 CPU）
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA is not available or not being used. Falling back to CPU.")
    print(f"Using device: {device}")  # 打印设备信息


    # 初始化 WandB 进行实验追踪
    run = initialize_wandb(args)

    # 创建并重置 Atari 环境
    env = create_atari_env(env_id)  # 使用自定义函数创建环境
    env.reset(seed=args.seed)  # 用指定种子重置环境

    # 初始化 PPO 模型
    model = PPO(policy_type, env, verbose=1, tensorboard_log=f"runs/ppo")  # 定义 PPO 算法实例

    '''
    在 PPO 初始化代码中，参数 verbose 用于控制日志输出的详细程度。它通常取以下值：
    verbose=0: 不输出任何日志信息。
    verbose=1: 输出基本的日志信息（通常是训练过程中的关键步骤或重要信息）。
    verbose=2: 输出更详细的日志信息，包括调试信息和更多过程细节。
    '''
    # 开始训练模型
    model.learn(total_timesteps=total_timesteps)  # 使用指定时间步数训练模型

    # 结束 WandB 跟踪并保存最终模型
    run.finish()

# 如果文件作为主程序运行，开始执行以下代码
if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Atari PPO Training')

    # 定义各种命令行参数
    #parser.add_argument('--env', type=str, default="ALE/Breakout-v5", help='Atari environment name')  # 环境名称
    parser.add_argument('--env', type=str, default="ALE/SpaceInvaders-v5", help='Atari environment name')  # 环境名称
    parser.add_argument('--policy_type', type=str, default="MlpPolicy", help='Policy type for PPO')  # PPO 策略类型
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='Total timesteps for training')  # 总时间步
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')  # 随机种子
    parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use CUDA for training')  # 是否使用 CUDA

    args = parser.parse_args()  # 解析命令行参数
    # 记录训练开始时间并运行主函数
    training_start_time = time.time()

    main(args)
    training_duration = time.time() - training_start_time  # 计算训练持续时间
    print(f'Training time: {training_duration / 3600:.2f} hours')  # 以小时为单位打印训练时间