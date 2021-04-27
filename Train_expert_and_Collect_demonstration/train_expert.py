import os
import argparse
from datetime import datetime
import torch
import gym
import sys
import numpy as np
from colabgymrender.recorder import Recorder
from sac_algo import SAC
from gym import wrappers
sys.path.insert(0,"/root/AIRL")
from trainer import Trainer
from env import make_env
from env_process import process_state_image

def run(args):
    # env = make_env(args.env_id)
    #env_test = make_env(args.env_id)
    #s = env.reset()
    env = gym.make('Pendulum-v0')
    env_test = gym.make('Pendulum-v0')
    directory = './video'
    env = wrappers.Monitor(env, directory, video_callable=False ,force=True)
    env_test = wrappers.Monitor(env_test, directory, video_callable=False ,force=True)
    # print("OBS_SHAPE",env.observation_shape)
    state = env.reset()
    # state = process_state_image(state)
    # print("STATE_SHAPE",env.observation_space.shape)
    # print("ACTION_SHAPE",env.action_space.shape)
    algo = SAC(
        #state_shape=env.observation_shape,
        #action_shape= env.action_space.shape,
        state_shape=env.observation_space.shape,
        action_shape= env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed
    )
    time = datetime.now().strftime("%Y%m%d-%H%M")
    # log_dir = os.path.join(
    #     '/gdrive/MyDrive/Colab Notebooks/AIRL/demonstration/logs', args.env_id, 'sac', f'seed{args.seed}-{time}')
    log_dir = os.path.join('logs', args.env_id, 'sac', f'seed{args.seed}-{time}')
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
    p.add_argument('--num_steps', type=int, default=10**6)
    p.add_argument('--eval_interval', type=int, default=10**4)
    p.add_argument('--env_id', type=str, default='Pendulum-v0')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)