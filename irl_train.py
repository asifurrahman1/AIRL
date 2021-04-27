import os
import argparse
from datetime import datetime
import torch
import gym
from roll_buffer import SerializedBuffer
from airl import AIRL
from trainer import Trainer


def run(args):
    # env = make_env(args.env_id)
    env = gym.make('Pendulum-v0')
    env_test = gym.make('Pendulum-v0')
    buffer_exp = SerializedBuffer(
        path=args.buffer,
        device=torch.device("cuda" if args.cuda else "cpu")
    )
    print(env.observation_space.shape)
    print(env.action_space.shape)
    algo = AIRL(
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'IRL_log', args.env_id, args.algo, f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, default='/gdrive/MyDrive/Colab Notebooks/AIRL_data/logs/Pendulum-v0/sac/seed0-20210422-1908/Expert_demonstration/size1000000_std0.0_prand0.0.pth')
    p.add_argument('--rollout_length', type=int, default=50000)
    p.add_argument('--num_steps', type=int, default=10**7)
    p.add_argument('--eval_interval', type=int, default=10**5)
    p.add_argument('--env_id', type=str, default='Pendulum-v0')
    p.add_argument('--algo', type=str, default='airl')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)