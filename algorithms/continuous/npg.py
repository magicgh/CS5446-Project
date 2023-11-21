import os
import fire
import torch
import pprint
import numpy as np

import gymnasium as gym
from tianshou.policy import NPGPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.continuous import ActorProb, Critic

task = "Pendulum-v1"
default_reward_threshold = {"Pendulum-v1": -250}


def npg(seed=1, buffer_size=50000, lr=1e-3,
        gamma=0.95, epoch=5, step_per_epoch=50000, step_per_collect=2048,
        repeat_per_collect=2, batch_size=99999, hidden_sizes=[64, 64],
        training_num=16, test_num=10, logdir="log", render=0.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        gae_lambda=0.95, rew_norm=1, norm_adv=1, optim_critic_iters=5,
        actor_step_size=0.5):
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
        hidden_sizes=hidden_sizes,
        activation=torch.nn.Tanh,
        device=device,
    )
    actor = ActorProb(net, action_shape, unbounded=True,
                      device=device).to(device)
    critic = Critic(
        Net(
            state_shape,
            hidden_sizes=hidden_sizes,
            device=device,
            activation=torch.nn.Tanh,
        ),
        device=device,
    ).to(device)

    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(critic.parameters(), lr=lr)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = NPGPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=gamma,
        reward_normalization=rew_norm,
        advantage_normalization=norm_adv,
        gae_lambda=gae_lambda,
        action_space=env.action_space,
        optim_critic_iters=optim_critic_iters,
        actor_step_size=actor_step_size,
        deterministic_eval=True,
    )

    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(buffer_size, len(train_envs)),
    )
    test_collector = Collector(policy, test_envs)

    log_path = os.path.join(logdir, task, "npg")
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
    fire.Fire(npg)
