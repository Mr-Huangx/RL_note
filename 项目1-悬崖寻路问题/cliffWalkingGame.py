import gym
import numpy as np
import matplotlib.pyplot as plt


# 1. 配置参数类
class Config:
    def __init__(self):
        self.env_name = 'CliffWalking-v0'
        self.policy_lr = 0.1  # 学习率 alpha
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 0.01  # epsilon-greedy策略的探索率
        self.train_eps = 500  # 训练的回合数


# 2. 环境包装器 (Wrapper)
# 作用：统一接口，确保由 gym 版本差异导致的数据格式问题得到解决
class CliffWalkingWapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        # 处理 Gym 新旧版本 step 返回值的差异
        # 旧版: next_state, reward, done, info
        # 新版: next_state, reward, terminated, truncated, info
        step_result = self.env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, info = step_result
        return next_state, reward, done, info

    def reset(self):
        # 处理 Gym 新旧版本 reset 返回值的差异
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            return reset_result[0]  # 新版返回 (state, info)
        return reset_result  # 旧版返回 state


# 3. Q-Learning 智能体算法
class QLearning:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon=0.1):
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        # 初始化 Q 表，大小为 [state_dim, action_dim]，初始值为 0
        self.Q = np.zeros((state_dim, action_dim))

    def choose_action(self, state):
        # Epsilon-Greedy 策略
        # 以 epsilon 的概率随机探索
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.action_dim)
        else:
            # 否则选择 Q 值最大的动作
            # 注意：如果多个动作Q值相同，np.argmax只返回第一个，这里可以加一点随机性打乱，
            # 但为了简化，直接用 argmax 即可
            action = np.argmax(self.Q[state, :])
        return action

    def update(self, state, action, reward, next_state, done):
        # Q-Learning 更新公式
        # Q(s,a) = Q(s,a) + lr * [R + gamma * max(Q(s', a')) - Q(s,a)]

        predict_Q = self.Q[state, action]

        if done:
            # 如果到达终点或掉入悬崖，没有下一个状态的价值
            target_Q = reward
        else:
            # 下一个状态的最大 Q 值 (Off-policy, 永远选最好的)
            target_Q = reward + self.gamma * np.max(self.Q[next_state, :])

        # 更新 Q 表
        self.Q[state, action] += self.lr * (target_Q - predict_Q)


# 4. 主训练循环 (你提供的逻辑)
def train(cfg):
    # 初始化环境
    try:
        env = gym.make(cfg.env_name, render_mode="rgb_array")  # render_mode 适配新版gym
    except:
        env = gym.make(cfg.env_name)  # 适配旧版gym

    env = CliffWalkingWapper(env)

    # 初始化智能体
    agent = QLearning(
        state_dim=env.observation_space.n,
        action_dim=env.action_space.n,
        learning_rate=cfg.policy_lr,
        gamma=cfg.gamma,
        epsilon=cfg.epsilon
    )

    rewards = []
    ma_rewards = []  # moving average reward

    for i_ep in range(cfg.train_eps):  # train_eps: 训练的最大episodes数
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset()  # 重置环境

        while True:
            action = agent.choose_action(state)  # 根据算法选择一个动作
            next_state, reward, done, _ = env.step(action)  # 与环境进行一次动作交互

            # 这里有一个小技巧：悬崖寻路默认掉坑里是 -100，每走一步是 -1
            # 原始环境不需要改动，直接传给 agent 更新即可
            agent.update(state, action, reward, next_state, done)  # Q-learning算法更新

            state = next_state  # 存储上一个观察值
            ep_reward += reward
            if done:
                break

        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)

        if (i_ep + 1) % 20 == 0:
            print("Episode:{}/{}: reward:{:.1f}".format(i_ep + 1, cfg.train_eps, ep_reward))

    return rewards, ma_rewards


# 5. 运行与绘图
if __name__ == "__main__":
    cfg = Config()
    rewards, ma_rewards = train(cfg)

    # 绘制结果
    plt.figure(figsize=(8, 5))
    plt.title("Q-Learning on CliffWalking")
    plt.plot(rewards, label='Rewards', color='cyan', alpha=0.6)
    plt.plot(ma_rewards, label='Moving Average Rewards', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()