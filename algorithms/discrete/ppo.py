import os
import fire
import torch
import pprint
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from tianshou.policy import PPOPolicy
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.common import ActorCritic, Net, DataParallelNet

task = "CartPole-v1"
default_reward_threshold = {"CartPole-v1": 495}


def ppo(seed=1626, buffer_size=20000, lr=3e-4,
        gamma=0.99, epoch=10, step_per_epoch=50000, step_per_collect=2000,
        repeat_per_collect=10, batch_size=64, hidden_sizes=[64, 64],
        training_num=20, test_num=100, logdir="log", render=0.0,
        device="cuda" if torch.cuda.is_available() else "cpu", vf_coef=0.5,
        ent_coef=0.0, eps_clip=0.2, max_grad_norm=0.5, gae_lambda=0.95,
        rew_norm=0, norm_adv=0, recompute_adv=0, dual_clip=None, value_clip=0):

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
    if torch.cuda.is_available():
        actor = DataParallelNet(
            Actor(net, action_shape, device=None).to(device))
        critic = DataParallelNet(Critic(net, device=None).to(device))
    else:
        actor = Actor(net, action_shape, device=device).to(device)
        critic = Critic(net, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)

    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)
    dist = torch.distributions.Categorical
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        action_scaling=isinstance(env.action_space, Box),
        discount_factor=gamma,
        max_grad_norm=max_grad_norm,
        eps_clip=eps_clip,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        gae_lambda=gae_lambda,
        reward_normalization=rew_norm,
        dual_clip=dual_clip,
        value_clip=value_clip,
        action_space=env.action_space,
        deterministic_eval=True,
        advantage_normalization=norm_adv,
        recompute_advantage=recompute_adv,
    )

    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(buffer_size, len(train_envs)),
    )
    test_collector = Collector(policy, test_envs)

    log_path = os.path.join(logdir, task, "ppo")
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
        step_per_collect=step_per_collect,
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
    fire.Fire(ppo)
