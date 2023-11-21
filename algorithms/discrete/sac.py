import os
import fire
import torch
import pprint
import numpy as np
import gymnasium as gym
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils import TensorboardLogger
from tianshou.trainer import OffpolicyTrainer
from tianshou.policy import DiscreteSACPolicy
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.data import Collector, VectorReplayBuffer


task = "CartPole-v1"
default_reward_threshold = {"CartPole-v1": 495}


def sac(seed=1, buffer_size=20000,
        actor_lr=1e-4, critic_lr=1e-3, alpha_lr=3e-4, gamma=0.95, tau=0.005,
        alpha=0.05, auto_alpha=False, epoch=5, step_per_epoch=10000,
        step_per_collect=10, update_per_step=0.1, batch_size=64,
        hidden_sizes=[64, 64], training_num=10, test_num=100, logdir="log",
        render=0.0, n_step=3, device="cuda" if torch.cuda.is_available() else "cpu"):
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

    net = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = Actor(net, action_shape, softmax_output=False,
                  device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    net_c1 = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    critic1 = Critic(net_c1, last_size=action_shape, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
    net_c2 = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    critic2 = Critic(net_c2, last_size=action_shape, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

    if auto_alpha:
        target_entropy = 0.98 * np.log(np.prod(action_shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)

    policy = DiscreteSACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        action_space=env.action_space,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=tau,
        gamma=gamma,
        alpha=alpha,
        estimation_step=n_step,
    )

    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(buffer_size, len(train_envs)),
    )
    test_collector = Collector(policy, test_envs)

    log_path = os.path.join(logdir, task, "discrete_sac")
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
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=update_per_step,
        test_in_train=False,
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
    fire.Fire(sac)
