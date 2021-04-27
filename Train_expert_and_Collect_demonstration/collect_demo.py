import os
from tqdm import tqdm
import argparse
import torch
import sys
import numpy as np
from sac_utils import add_random_noise
from sac_algo import SACExpert
sys.path.insert(0,"/root/AIRL")
from buffer_sac import Buffer
import gym


def collect_demo(env, algo, buffer_size, device, std, p_rand, seed=0):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )
    total_return = 0.0
    num_episodes = 0

    state = env.reset()
    t = 0
    episode_return = 0.0
    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1
        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            action = algo.exploit(state)
            action = add_random_noise(action, std)

        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        buffer.append(state, action, reward, mask, next_state)
        episode_return += reward

        if done:
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0

        state = next_state
    print(f'Mean return of the expert is {total_return / num_episodes}')
    return buffer

def run(args):
    env = gym.make('Pendulum-v0')
    print(env.observation_space.shape)
    print(env.action_space.shape)
    algo = SACExpert(
        state_shape=env.observation_space.shape,
        action_shape= env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        path=args.weight
    )
    buffer = collect_demo(
        env=env,
        algo=algo,
        buffer_size=args.buffer_size,
        device=torch.device("cuda" if args.cuda else "cpu"),
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed
    )
    buffer.save(os.path.join(
        '/gdrive/MyDrive/Colab Notebooks/AIRL_data/logs/Pendulum-v0/sac/seed0-20210422-1908/Expert_demonstration/',
        f'size{args.buffer_size}_std{args.std}_prand{args.p_rand}.pth'
    ))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weight', type=str, default='/gdrive/MyDrive/Colab Notebooks/AIRL_data/logs/Pendulum-v0/sac/seed0-20210422-1908/model/step1000000/actor.pth')
    p.add_argument('--env_id', type=str, default='Pendulum-v0')
    p.add_argument('--buffer_size', type=int, default=10**6)
    p.add_argument('--std', type=float, default=0.0)
    p.add_argument('--p_rand', type=float, default=0.0)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)