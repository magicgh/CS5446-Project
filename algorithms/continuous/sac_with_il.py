import os
import fire
import torch
import pprint
import envpool
import numpy as np
import gymnasium as gym
from tianshou.utils.net.common import Net
from tianshou.utils import TensorboardLogger
from tianshou.trainer import OffpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.continuous import ActorProb, Critic, Actor
from tianshou.policy import SACPolicy, ImitationPolicy

task = "Pendulum-v1"
default_reward_threshold = {"Pendulum-v1": -250}


def sac_with_il(seed=0, buffer_size=20000,
                hidden_sizes=[128, 128], actor_lr=1e-3, critic_lr=1e-3, il_lr=1e-3, epoch=7,
                step_per_epoch=24000, batch_size=256, training_num=10, test_num=10,
                step_per_collect=10, update_per_step=0.1, logdir="log",
                gamma=0.99, tau=0.005, device="cuda" if torch.cuda.is_available() else "cpu",
                alpha=0.2, auto_alpha=1, alpha_lr=3e-4, il_step_per_epoch=500,
                imitation_hidden_sizes=[128, 128], render=0.0, n_step=3):
    train_envs = env = envpool.make_gymnasium(
        task, num_envs=training_num, seed=seed)
    test_envs = envpool.make_gymnasium(task, num_envs=test_num, seed=seed)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    reward_threshold = default_reward_threshold.get(
        task, env.spec.reward_threshold)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = ActorProb(net, action_shape, device=device,
                      unbounded=True).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    net_c1 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
    net_c2 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

    if auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=tau,
        gamma=gamma,
        alpha=alpha,
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
    log_path = os.path.join(logdir, task, "sac")
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

    policy.eval()
    net = Actor(
        Net(
            state_shape,
            hidden_sizes=imitation_hidden_sizes,
            device=device,
        ),
        action_shape,
        max_action=max_action,
        device=device,
    ).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=il_lr)
    il_policy = ImitationPolicy(
        actor=net,
        optim=optim,
        action_space=env.action_space,
        action_scaling=True,
        action_bound_method="clip",
    )
    il_test_collector = Collector(
        il_policy,
        envpool.make_gymnasium(task, num_envs=test_num, seed=seed),
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
        policy.eval()
        il_test_collector.reset()
        result = il_test_collector.collect(n_episode=1, render=render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    fire.Fire(sac_with_il)
