import os
import fire
import torch
import pprint
import pickle
import numpy as np
import gymnasium as gym
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.trainer import OffpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.policy import GAILPolicy, SACPolicy
from torch.distributions import Independent, Normal
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.continuous import ActorProb, Critic


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


def gail(seed=1, buffer_size=20000,
         lr=1e-3, disc_lr=5e-4, gamma=0.95,
         epoch=5, step_per_epoch=20000, episode_per_collect=16,
         repeat_per_collect=2, batch_size=128, disc_update_num=2,
         hidden_sizes=[128, 128], training_num=8, test_num=100,
         logdir="log", render=0.0, n_step=3,
         device="cuda" if torch.cuda.is_available() else "cpu",
         vf_coef=0.25, ent_coef=0.0, eps_clip=0.2, max_grad_norm=0.5,
         gae_lambda=0.95, rew_norm=1, dual_clip=None, value_clip=1, norm_adv=1,
         recompute_adv=0, resume=False, save_interval=4, load_buffer_name=expert_file_name):
    if os.path.exists(load_buffer_name) and os.path.isfile(load_buffer_name):
        if load_buffer_name.endswith(".hdf5"):
            buffer = VectorReplayBuffer.load_hdf5(load_buffer_name)
        else:
            with open(load_buffer_name, "rb") as f:
                buffer = pickle.load(f)
    else:
        buffer = gather_data()
    env = gym.make(task)
    reward_threshold = default_reward_threshold.get(
        task, env.spec.reward_threshold)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    print(env.action_space)
    max_action = env.action_space.high[0]
    train_envs = DummyVectorEnv([lambda: gym.make(task)
                                for _ in range(training_num)])
    test_envs = DummyVectorEnv([lambda: gym.make(task)
                               for _ in range(test_num)])
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)
    net = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = ActorProb(net, action_shape, max_action=max_action, device=device).to(
        device,
    )
    critic = Critic(
        Net(state_shape, hidden_sizes=hidden_sizes, device=device),
        device=device,
    ).to(device)
    actor_critic = ActorCritic(actor, critic)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)
    disc_net = Critic(
        Net(
            state_shape,
            action_shape=action_shape,
            hidden_sizes=hidden_sizes,
            activation=torch.nn.Tanh,
            device=device,
            concat=True,
        ),
        device=device,
    ).to(device)
    for m in disc_net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    disc_optim = torch.optim.Adam(disc_net.parameters(), lr=disc_lr)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = GAILPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        expert_buffer=buffer,
        disc_net=disc_net,
        disc_optim=disc_optim,
        disc_update_num=disc_update_num,
        discount_factor=gamma,
        max_grad_norm=max_grad_norm,
        eps_clip=eps_clip,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        reward_normalization=rew_norm,
        advantage_normalization=norm_adv,
        recompute_advantage=recompute_adv,
        dual_clip=dual_clip,
        value_clip=value_clip,
        gae_lambda=gae_lambda,
        action_space=env.action_space,
    )
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(buffer_size, len(train_envs)),
    )
    test_collector = Collector(policy, test_envs)
    log_path = os.path.join(logdir, task, "gail")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, save_interval=save_interval)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return mean_rewards >= reward_threshold

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        torch.save(
            {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
            },
            ckpt_path,
        )
        return ckpt_path

    if resume:
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            policy.load_state_dict(checkpoint["model"])
            optim.load_state_dict(checkpoint["optim"])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")

    # trainer
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
        resume_from_log=resume,
        save_checkpoint_fn=save_checkpoint_fn,
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
    fire.Fire(gail)
