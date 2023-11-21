import os
import fire
import torch
import pprint
import pickle
import datetime
import numpy as np
import gymnasium as gym
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils import TensorboardLogger
from tianshou.exploration import GaussianNoise
from torch.utils.tensorboard import SummaryWriter
from tianshou.policy import TD3BCPolicy, SACPolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OffpolicyTrainer, OfflineTrainer
from tianshou.utils.net.continuous import ActorProb, Critic, Actor

task = "Pendulum-v1"
default_reward_threshold = {"Pendulum-v1": -250}


def expert_file_name():
    return os.path.join(os.path.dirname(__file__), "expert_SAC_Pendulum-v1.pkl")


def gather_data(seed=0, buffer_size=20000,
                hidden_sizes=[128, 128], actor_lr=1e-3, critic_lr=1e-3, epoch=7,
                step_per_epoch=8000, batch_size=256, training_num=10, test_num=10,
                step_per_collect=10, update_per_step=0.125, logdir="log",
                gamma=0.99, tau=0.005, device="cuda" if torch.cuda.is_available() else "cpu",
                alpha=0.2, auto_alpha=1, alpha_lr=3e-4,
                n_step=3, save_buffer_name=expert_file_name()):
    """Return expert buffer data."""

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
    actor = ActorProb(
        net,
        action_shape,
        device=device,
        unbounded=True,
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    net_c = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    critic = Critic(net_c, device=device).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    if auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        tau=tau,
        gamma=gamma,
        alpha=alpha,
        estimation_step=n_step,
        action_space=env.action_space,
    )
    buffer = VectorReplayBuffer(buffer_size, len(train_envs))
    train_collector = Collector(
        policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    log_path = os.path.join(logdir, task, "sac")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return mean_rewards >= reward_threshold

    OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        episode_per_test=test_num,
        batch_size=batch_size,
        update_per_step=update_per_step,
        save_best_fn=save_best_fn,
        stop_fn=stop_fn,
        logger=logger,
    ).run()
    train_collector.reset()
    result = train_collector.collect(n_step=buffer_size)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
    if save_buffer_name.endswith(".hdf5"):
        buffer.save_hdf5(save_buffer_name)
    else:
        with open(save_buffer_name, "wb") as f:
            pickle.dump(buffer, f)
    return buffer


def td3_bc(seed=0, hidden_sizes=[64, 64],
           actor_lr=1e-3, critic_lr=1e-3, epoch=5, step_per_epoch=500, n_step=3,
           batch_size=64, alpha=2.5, exploration_noise=0.1, policy_noise=0.2,
           noise_clip=0.5, update_actor_freq=2, tau=0.005, gamma=0.99,
           test_num=10, logdir="log", render=1/35,
           device="cuda" if torch.cuda.is_available() else "cpu", resume_path=None,
           load_buffer_name=expert_file_name()):
    if os.path.exists(load_buffer_name) and os.path.isfile(load_buffer_name):
        if load_buffer_name.endswith(".hdf5"):
            buffer = VectorReplayBuffer.load_hdf5(load_buffer_name)
        else:
            with open(load_buffer_name, "rb") as f:
                buffer = pickle.load(f)
    else:
        buffer = gather_data()
    env = gym.make(task)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    reward_threshold = default_reward_threshold.get(
        task, env.spec.reward_threshold)

    test_envs = DummyVectorEnv([lambda: gym.make(task)
                               for _ in range(test_num)])

    np.random.seed(seed)
    torch.manual_seed(seed)
    test_envs.seed(seed)

    net_a = Net(
        state_shape,
        hidden_sizes=hidden_sizes,
        device=device,
    )
    actor = Actor(
        net_a,
        action_shape=action_shape,
        max_action=max_action,
        device=device,
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)

    net_c1 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    net_c2 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

    policy = TD3BCPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=tau,
        gamma=gamma,
        exploration_noise=GaussianNoise(sigma=exploration_noise),
        policy_noise=policy_noise,
        update_actor_freq=update_actor_freq,
        noise_clip=noise_clip,
        alpha=alpha,
        estimation_step=n_step,
        action_space=env.action_space,
    )

    if resume_path:
        policy.load_state_dict(torch.load(resume_path, map_location=device))
        print("Loaded agent from: ", resume_path)

    test_collector = Collector(policy, test_envs)

    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{seed}_{t0}-{task.replace("-", "_")}_td3_bc'
    log_path = os.path.join(logdir, task, "td3_bc", log_file)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return mean_rewards >= reward_threshold

    trainer = OfflineTrainer(
        policy=policy,
        buffer=buffer,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        episode_per_test=test_num,
        batch_size=batch_size,
        save_best_fn=save_best_fn,
        stop_fn=stop_fn,
        logger=logger,
    )

    for epoch, epoch_stat, info in trainer:
        print(f"Epoch: {epoch}")
        print(epoch_stat)
        print(info)

    assert stop_fn(info["best_reward"])

    if __name__ == "__main__":
        pprint.pprint(info)
        env = gym.make(task)
        policy.eval()
        collector = Collector(policy, env)
        collector_result = collector.collect(n_episode=1, render=render)
        rews, lens = collector_result["rews"], collector_result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    fire.Fire(td3_bc)
