import os
import fire
import torch
import pprint
import numpy as np
import gymnasium as gym
from tianshou.policy import DDPGPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils import TensorboardLogger
from tianshou.trainer import OffpolicyTrainer
from tianshou.exploration import GaussianNoise
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.continuous import Critic, Actor


task = "Pendulum-v1"
default_reward_threshold = {"Pendulum-v1": -250}


def ddpg(seed=0, buffer_size=20000,
         actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005, exploration_noise=0.1,
         epoch=5, step_per_epoch=20000, step_per_collect=8, update_per_step=0.125,
         batch_size=128, hidden_sizes=[128, 128], training_num=8, test_num=100,
         logdir="log", render=0.0, n_step=3,
         device="cuda" if torch.cuda.is_available() else "cpu"):
    env = gym.make(task)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
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
    net = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = Actor(net, action_shape, max_action=max_action, device=device).to(
        device,
    )
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    net = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    critic = Critic(net, device=device).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    policy = DDPGPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        tau=tau,
        gamma=gamma,
        exploration_noise=GaussianNoise(sigma=exploration_noise),
        estimation_step=n_step,
        action_space=env.action_space,
    )
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs)
    log_path = os.path.join(logdir, task, "ddpg")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return mean_rewards >= reward_threshold

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
    fire.Fire(ddpg)
