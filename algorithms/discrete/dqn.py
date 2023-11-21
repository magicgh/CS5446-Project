import os
import fire
import torch
import pprint
import numpy as np
import gymnasium as gym
from tianshou.policy import DQNPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils import TensorboardLogger
from tianshou.trainer import OffpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer


task = "CartPole-v1"
default_reward_threshold = {"CartPole-v1": 495}


def dqn(seed=1626, eps_test=0.05, eps_train=0.1,
        buffer_size=20000, lr=1e-3, gamma=0.9, n_step=3, target_update_freq=320,
        epoch=20, step_per_epoch=10000, step_per_collect=10, update_per_step=0.1,
        batch_size=64, hidden_sizes=[128, 128, 128, 128], training_num=10, test_num=100,
        logdir="log", render=0.0, prioritized_replay=False, alpha=0.6, beta=0.4,
        device="cuda" if torch.cuda.is_available() else "cpu"):
    env = gym.make(task)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    reward_threshold = default_reward_threshold.get(
        task, env.spec.reward_threshold)
    train_envs = DummyVectorEnv([lambda: gym.make(task)
                                for _ in range(training_num)])
    test_envs = DummyVectorEnv([lambda: gym.make(task)
                               for _ in range(test_num)])
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    net = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        device=device,
    ).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    policy = DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=gamma,
        estimation_step=n_step,
        target_update_freq=target_update_freq,
        action_space=env.action_space,
    )

    if prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            buffer_size,
            buffer_num=len(train_envs),
            alpha=alpha,
            beta=beta,
        )
    else:
        buf = VectorReplayBuffer(buffer_size, buffer_num=len(train_envs))
    train_collector = Collector(
        policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    train_collector.collect(n_step=batch_size * training_num)
    log_path = os.path.join(logdir, task, "dqn")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return mean_rewards >= reward_threshold

    def train_fn(epoch, env_step):
        if env_step <= 10000:
            policy.set_eps(eps_train)
        elif env_step <= 50000:
            eps = eps_train - (env_step - 10000) / 40000 * (0.9 * eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * eps_train)

    def test_fn(epoch, env_step):
        policy.set_eps(eps_test)

    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        episode_per_test=test_num,
        batch_size=batch_size,
        update_per_step=update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()
    assert stop_fn(result["best_reward"])

    if __name__ == "__main__":
        pprint.pprint(result)
        env = gym.make(task)
        policy.eval()
        policy.set_eps(eps_test)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    fire.Fire(dqn)
