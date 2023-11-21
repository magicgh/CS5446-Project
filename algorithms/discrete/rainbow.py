import os
import fire
import torch
import pickle
import pprint
import numpy as np
import gymnasium as gym
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.policy import RainbowPolicy
from tianshou.utils import TensorboardLogger
from tianshou.trainer import OffpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.net.discrete import NoisyLinear
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer


task = "CartPole-v1"
default_reward_threshold = {"CartPole-v1": 495}


def rainbow(seed=1626, eps_test=0.05, eps_train=0.1,
            buffer_size=20000, lr=1e-3, gamma=0.9, num_atoms=51, v_min=-10.0, v_max=10.0,
            noisy_std=0.1, n_step=3, target_update_freq=320, epoch=10, step_per_epoch=8000,
            step_per_collect=8, update_per_step=0.125, batch_size=64,
            hidden_sizes=[128, 128, 128, 128], training_num=8, test_num=100,
            logdir="log", render=0.0, prioritized_replay=False, alpha=0.6, beta=0.4,
            beta_final=1.0, resume=False, device="cuda" if torch.cuda.is_available() else "cpu",
            save_interval=4):
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

    def noisy_linear(x, y):
        return NoisyLinear(x, y, noisy_std)

    net = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        device=device,
        softmax=True,
        num_atoms=num_atoms,
        dueling_param=({"linear_layer": noisy_linear},
                       {"linear_layer": noisy_linear}),
    )
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    policy = RainbowPolicy(
        model=net,
        optim=optim,
        discount_factor=gamma,
        action_space=env.action_space,
        num_atoms=num_atoms,
        v_min=v_min,
        v_max=v_max,
        estimation_step=n_step,
        target_update_freq=target_update_freq,
    ).to(device)

    if prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            buffer_size,
            buffer_num=len(train_envs),
            alpha=alpha,
            beta=beta,
            weight_norm=True,
        )
    else:
        buf = VectorReplayBuffer(buffer_size, buffer_num=len(train_envs))

    train_collector = Collector(
        policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    train_collector.collect(n_step=batch_size * training_num)

    log_path = os.path.join(logdir, task, "rainbow")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, save_interval=save_interval)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return mean_rewards >= reward_threshold

    def train_fn(epoch, env_step):

        if env_step <= 10000:
            policy.set_eps(eps_train)
        elif env_step <= 50000:
            eps = eps_train - (env_step - 10000) / 40000 * (0.9 * eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * eps_train)

        if prioritized_replay:
            if env_step <= 10000:
                beta = beta
            elif env_step <= 50000:
                beta = beta - (env_step - 10000) / 40000 * (beta - beta_final)
            else:
                beta = beta_final
            buf.set_beta(beta)

    def test_fn(epoch, env_step):
        policy.set_eps(eps_test)

    def save_checkpoint_fn(epoch, env_step, gradient_step):

        ckpt_path = os.path.join(log_path, "checkpoint.pth")

        torch.save(
            {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
            },
            ckpt_path,
        )
        buffer_path = os.path.join(log_path, "train_buffer.pkl")
        with open(buffer_path, "wb") as f:
            pickle.dump(train_collector.buffer, f)
        return ckpt_path

    if resume:

        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            policy.load_state_dict(checkpoint["model"])
            policy.optim.load_state_dict(checkpoint["optim"])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")
        buffer_path = os.path.join(log_path, "train_buffer.pkl")
        if os.path.exists(buffer_path):
            with open(buffer_path, "rb") as f:
                train_collector.buffer = pickle.load(f)
            print("Successfully restore buffer.")
        else:
            print("Fail to restore buffer.")

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
        train_fn=train_fn,
        test_fn=test_fn,
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
        policy.set_eps(eps_test)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    fire.Fire(rainbow)
