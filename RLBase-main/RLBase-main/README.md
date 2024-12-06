# Reinforcement Learning with PPO on Atari Games

This project implements a reinforcement learning (RL) setup for training agents using the **Proximal Policy Optimization (PPO)** algorithm on Atari environments. The implementation uses **Stable-Baselines3** for PPO and **Gym** for Atari environments. The training process is logged with **Weights & Biases (WandB)** for experiment tracking.

## Project Structure

```
├── algo/
│   ├── ppo/               # PPO algorithm implementation
│   ├── common/            # Common utility functions for RL
│   ├── ...                # Other RL algorithms (e.g., DQN, A2C, etc.)
├── envs/
│   └── atari_env.py       # Atari environment setup
├── main.py                # Main entry point for training
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── runs/                  # Directory to store training logs
├── wandb/                 # Directory for WandB logs
└── ...                    # Additional files and directories
```

## Requirements

- Python 3.7+
- PyTorch 1.10+
- Gym
- Stable-Baselines3
- Weights & Biases
- Other dependencies listed in `requirements.txt`

### Install Dependencies

To install the required packages, run the following command:

```bash
conda create -n rlbase python==3.10
conda activate rlbase
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
```

## Usage

### Training the Model

To start training a PPO agent on an Atari environment, run the following command:

```bash
python main.py --env ALE/Breakout-v5 --policy_type CnnPolicy --total_timesteps 1000000 --use_cuda True
```

#### Command-Line Arguments

- `--env`: The Atari environment to train on (default: `ALE/Breakout-v5`).
- `--policy_type`: The type of policy to use (default: `CnnPolicy`).
- `--total_timesteps`: The number of timesteps for training (default: `1000000`).
- `--seed`: Random seed for reproducibility (optional).
- `--use_cuda`: Whether to use CUDA for training (default: `True`).

### Experiment Logging with WandB

Training progress is automatically logged to [Weights & Biases](https://wandb.ai/) for experiment tracking. Ensure that your WandB account is connected and properly set up. Logs include training metrics, configurations, and model checkpoints.

## File Descriptions

- `main.py`: The main script to initialize and run the training process.
- `algo/ppo/ppo.py`: Implementation of the PPO algorithm.
- `envs/atari_env.py`: Helper functions to create Atari environments for RL tasks.
- `requirements.txt`: A list of required Python packages for the project.
- `wandb/`: Contains logs and data for experiment tracking.

## Notes

- This project supports training on **CUDA-enabled devices** for faster training times, but can also be run on a CPU.
- Ensure that your system has access to the required Atari ROMs (which are generally required by Gym for Atari environments).
- Experiment data is automatically synced to WandB, providing visualizations of the training process.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project uses the PPO algorithm from [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).
- Atari environment setup is based on the [Gym Atari](https://gym.openai.com/envs/Atari-Environment/) library.
- Logging and visualization via [Weights & Biases](https://wandb.ai/).


