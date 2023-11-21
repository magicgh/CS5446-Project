import os
import fire
import torch
import pprint
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from tianshou.policy import PGPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer


task = "CartPole-v1"
default_reward_threshold = {"CartPole-v1": 495}


def pg(seed=1, buffer_size=20000, lr=1e-3,
       gamma=0.95, epoch=10, step_per_epoch=40000, episode_per_collect=8,
       repeat_per_collect=2, batch_size=64, hidden_sizes=[64, 64],
       training_num=8, test_num=100, logdir="log", render=0.0, rew_norm=1,
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
        softmax=True,
    ).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    dist_fn = torch.distributions.Categorical
    policy = PGPolicy(
        actor=net,
        optim=optim,
        dist_fn=dist_fn,
        discount_factor=gamma,
        action_space=env.action_space,
        action_scaling=isinstance(env.action_space, Box),
        reward_normalization=rew_norm,
    )
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):

            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(buffer_size, len(train_envs)),
    )
    test_collector = Collector(policy, test_envs)

    log_path = os.path.join(logdir, task, "pg")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return mean_rewards >= reward_threshold

    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        repeat_per_collect=repeat_per_collect,
        episode_per_test=test_num,
        batch_size=batch_size,
        episode_per_collect=episode_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()
    assert stop_fn(result["best_reward"])

    if __name__ == "__main__":
        pprint.pprint(result)

        env = gym.make(task)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    fire.Fire(pg)
