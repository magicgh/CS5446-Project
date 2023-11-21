import os
import fire
import torch
import pprint
import envpool
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.policy import A2CPolicy, ImitationPolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OffpolicyTrainer, OnpolicyTrainer


task = "CartPole-v1"
default_reward_threshold = {"CartPole-v1": 495}


def a2c_with_il(seed=1, buffer_size=20000, lr=1e-3, il_lr=1e-3,
                gamma=0.9, epoch=10, step_per_epoch=50000, il_step_per_epoch=1000, episode_per_collect=16,
                step_per_collect=16, repeat_per_collect=1, batch_size=64,
                hidden_sizes=[64, 64], training_num=16, test_num=100, logdir="log",
                render=0.0, device="cuda" if torch.cuda.is_available() else "cpu", vf_coef=0.5, ent_coef=0.0,
                max_grad_norm=None, gae_lambda=1.0, rew_norm=False):

    train_envs = env = envpool.make(
        task,
        env_type="gymnasium",
        num_envs=training_num,
        seed=seed,
    )
    test_envs = envpool.make(
        task,
        env_type="gymnasium",
        num_envs=test_num,
        seed=seed,
    )
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    reward_threshold = default_reward_threshold.get(
        task, env.spec.reward_threshold)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = Actor(net, action_shape, device=device).to(device)
    critic = Critic(net, device=device).to(device)
    optim = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=lr)
    dist = torch.distributions.Categorical
    policy = A2CPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        action_scaling=isinstance(env.action_space, Box),
        discount_factor=gamma,
        gae_lambda=gae_lambda,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        reward_normalization=rew_norm,
        action_space=env.action_space,
    )

    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(buffer_size, len(train_envs)),
    )
    test_collector = Collector(policy, test_envs)

    log_path = os.path.join(logdir, task, "a2c")
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

    policy.eval()
    net = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    net = Actor(net, action_shape, device=device).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=il_lr)
    il_policy = ImitationPolicy(
        actor=net, optim=optim, action_space=env.action_space)
    il_test_collector = Collector(
        il_policy,
        envpool.make(task, env_type="gymnasium", num_envs=test_num, seed=seed),
    )
    train_collector.reset()
    result = OffpolicyTrainer(
        policy=il_policy,
        train_collector=train_collector,
        test_collector=il_test_collector,
        max_epoch=epoch,
        step_per_epoch=il_step_per_epoch,
        step_per_collect=step_per_collect,
        episode_per_test=test_num,
        batch_size=batch_size,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()
    assert stop_fn(result["best_reward"])

    if __name__ == "__main__":
        pprint.pprint(result)
        env = gym.make(task)
        il_policy.eval()
        collector = Collector(il_policy, env)
        result = collector.collect(n_episode=1, render=render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    fire.Fire(a2c_with_il)
