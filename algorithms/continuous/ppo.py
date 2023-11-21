import os
import fire
import torch
import pprint
import numpy as np
import gymnasium as gym
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.trainer import OnpolicyTrainer
from tianshou.policy import PPOPolicy

task = "Pendulum-v1"
default_reward_threshold = {"Pendulum-v1": -250}


def ppo(seed=1, buffer_size=20000, lr=1e-3,
        gamma=0.95, epoch=5, step_per_epoch=150000, episode_per_collect=16,
        repeat_per_collect=10, batch_size=64, hidden_sizes=[64, 64],
        training_num=20, test_num=100, logdir="log", render=0.0,
        device="cuda" if torch.cuda.is_available() else "cpu", vf_coef=0.5,
        ent_coef=0.0, eps_clip=0.2, max_grad_norm=0.5, gae_lambda=0.95,
        rew_norm=0, norm_adv=0, recompute_adv=0, dual_clip=None, value_clip=0,
        resume=False, save_interval=4):
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
    actor = ActorProb(net, action_shape, unbounded=True,
                      device=device).to(device)
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

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
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
    log_path = os.path.join(logdir, task, "ppo_con")
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

    trainer = OnpolicyTrainer(
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
        result = collector.collect(n_episode=1, render=render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    fire.Fire(ppo)
