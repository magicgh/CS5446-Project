import os
import fire
import torch
import pprint
import numpy as np
import gymnasium as gym
from tianshou.policy import FQFPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils import TensorboardLogger
from tianshou.trainer import OffpolicyTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.utils.net.discrete import FractionProposalNetwork, FullQuantileFunction


task = "CartPole-v1"
default_reward_threshold = {"CartPole-v1": 495}


def fqf(seed=1, eps_test=0.05, eps_train=0.1,
        buffer_size=20000, lr=3e-3, fraction_lr=2.5e-9, gamma=0.9, num_fractions=32,
        num_cosines=64, ent_coef=10.0, n_step=3, target_update_freq=320,
        epoch=10, step_per_epoch=10000, step_per_collect=10, update_per_step=0.1,
        batch_size=64, hidden_sizes=[64, 64, 64], training_num=10, test_num=100,
        logdir="log", render=0.0, prioritized_replay=False, alpha=0.6, beta=0.4,
        device="cuda" if torch.cuda.is_available() else "cpu"):
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

    feature_net = Net(
        state_shape,
        hidden_sizes[-1],
        hidden_sizes=hidden_sizes[:-1],
        device=device,
        softmax=False,
    )
    net = FullQuantileFunction(
        feature_net,
        action_shape,
        hidden_sizes,
        num_cosines=num_cosines,
        device=device,
    )
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    fraction_net = FractionProposalNetwork(num_fractions, net.input_dim)
    fraction_optim = torch.optim.RMSprop(
        fraction_net.parameters(), lr=fraction_lr)
    policy = FQFPolicy(
        model=net,
        optim=optim,
        fraction_model=fraction_net,
        fraction_optim=fraction_optim,
        action_space=env.action_space,
        discount_factor=gamma,
        num_fractions=num_fractions,
        ent_coef=ent_coef,
        estimation_step=n_step,
        target_update_freq=target_update_freq,
    ).to(device)

    if prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            buffer_size,
            buffer_num=len(train_envs),
            alpha=alpha,
            beta=beta,
        )
    else:
        buf = VectorReplayBuffer(buffer_size, buffer_num=len(train_envs))

    train_collector = Collector(
        policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    train_collector.collect(n_step=batch_size * training_num)

    log_path = os.path.join(logdir, task, "fqf")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

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

    def test_fn(epoch, env_step):
        policy.set_eps(eps_test)

    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        episode_per_test=test_num,
        batch_size=batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=update_per_step,
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
    fire.Fire(fqf)
