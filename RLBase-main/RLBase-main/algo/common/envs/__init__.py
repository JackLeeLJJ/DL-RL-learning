from common.envs.bit_flipping_env import BitFlippingEnv
from common.envs.identity_env import (
    FakeImageEnv,
    IdentityEnv,
    IdentityEnvBox,
    IdentityEnvMultiBinary,
    IdentityEnvMultiDiscrete,
)
from common.envs.multi_input_envs import SimpleMultiObsEnv

__all__ = [
    "BitFlippingEnv",
    "FakeImageEnv",
    "IdentityEnv",
    "IdentityEnvBox",
    "IdentityEnvMultiBinary",
    "IdentityEnvMultiDiscrete",
    "SimpleMultiObsEnv",
    "SimpleMultiObsEnv",
]


'''上面代码实现了多个环境，每个环境都用于不同的测试或模拟场景，以下是环境的总结及其作用：  

### 1. **BitFlippingEnv**
   - **作用**: 模拟一个简单的翻转位的环境，用于测试HER（Hindsight Experience Replay）等强化学习算法。
   - **特点**:
     - 目标是将所有位翻转为1。
     - 支持离散和连续动作空间。
     - 可选择使用离散观测空间或图像观测空间。

### 2. **IdentityEnv**
   - **作用**: 用于强化学习算法的简单测试，目标是输出和输入保持一致。
   - **特点**:
     - 支持不同的动作和观测空间（离散、连续、多离散、多二值）。
     - 可以控制每个episode的长度。

### 3. **IdentityEnvBox**
   - **作用**: 针对连续动作和观测空间的Identity环境测试。
   - **特点**:
     - 动作和状态为连续值。
     - 提供了一个精度参数`eps`，定义动作和状态相匹配的容忍范围。

### 4. **IdentityEnvMultiDiscrete**
   - **作用**: 针对多离散动作和观测空间的测试环境。
   - **特点**:
     - 适用于多维离散动作空间的测试。

### 5. **IdentityEnvMultiBinary**
   - **作用**: 针对多二值（MultiBinary）动作和观测空间的测试环境。
   - **特点**:
     - 用于强化学习算法在二值动作空间下的测试。

### 6. **FakeImageEnv**
   - **作用**: 模拟像Atari游戏一样的图像环境，用于强化学习图像处理能力的测试。
   - **特点**:
     - 动作空间可以是离散或连续。
     - 观测空间是灰度图像，支持通道优先或通道最后的图像格式。
     - 支持用户定义图像的高度、宽度和通道数。

### 7. **SimpleMultiObsEnv**
   - **作用**: 基于网格世界的多模态观测环境，用于测试强化学习算法在多种观测空间（向量和图像）下的表现。
   - **特点**:
     - 环境为4x4网格世界，部分状态不可访问。
     - 支持随机起点和多种动作空间（离散或连续）。
     - 每个状态通过随机生成的向量和图像表示。

---

### 总结
这些环境主要用于强化学习算法的开发、调试和测试，涵盖了不同的动作空间（离散、连续、多离散、多二值）和观测空间（标量、向量、图像等）。  
- 简单环境如`IdentityEnv`系列适合基础功能测试。  
- 复杂环境如`BitFlippingEnv`和`SimpleMultiObsEnv`则用于更高层次的功能验证和性能测试。  
- 图像相关环境如`FakeImageEnv`用于测试强化学习在处理高维输入（如图像）时的能力。'''