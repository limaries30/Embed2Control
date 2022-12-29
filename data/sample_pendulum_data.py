import os
from os import path
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import gym
import json
from datetime import datetime
import argparse
import yaml
from easydict import EasyDict as edict
from PIL import Image

# env = gym.make('Pendulum-v1', render_mode='rgb_array')
env = gym.make('Pendulum-v1')   # should use this to change gym.env state manually
env = env.unwrapped
env.reset()
env.state = np.array([np.pi*7/8, 0])
width, height = 48 * 2, 48


def rgb2bw(img):
    # thresh = 200
    pil_image = Image.fromarray(img)
    img_bw = pil_image.convert('L').resize((int(width/2), height)).point(lambda x: 0 if x < 128 else 255, '1')
    # option '1' in point() make image return bool value

    return np.array(img_bw)


def sample(sample_size):
    """
    return [(s, u, s_next)]
    """
    state_samples = []
    obs_samples = []
    for i in trange(sample_size, desc='Sampling data'):

        th = np.random.uniform(0, np.pi * 2)
        thdot = np.random.uniform(-8, 8)
        state = np.array([th, thdot])
        env.state = state
        u = np.random.uniform(-2, 2, size=(1,))
        _ = env.step(u)

        before1, before2 = render(state, u)

        after_state = env.state  # np.copy(state)
        after1, after2 = render(after_state, u)

        before = np.hstack((before1, before2))
        after = np.hstack((after1, after2))

        obs_samples.append((before, u, after))

    return obs_samples

def render(state, u):
    # need two observations to restore the Markov property

    env.state = state
    before1 = rgb2bw(env.render(mode='rgb_array'))  # return image of gym environment state
    # u = np.random.uniform(-2, 2, size=(1,))
    next_state, reward, done, info = env.step(u)
    before2 = rgb2bw(env.render(mode='rgb_array'))
    return (before1, before2)

def sample_pendulum(sample_size, output_dir="C:/Users/kihong/Documents/E2C_kihong/data/pendulum2", step_size=1, apply_control=True, num_shards=10):
    assert sample_size % num_shards == 0

    samples = []

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    for i in trange(sample_size):
        """
        for each sample:
        - draw a random state (theta, theta dot)
        - render x_t (including 2 images)
        - draw a random action u_t and apply
        - render x_t+1 after applying u_t
        """
        # th (theta) and thdot (theta dot) represent a state in Pendulum env
        th = np.random.uniform(0, np.pi*2)
        thdot = np.random.uniform(-8, 8)

        state = np.array([th, thdot])

        initial_state = np.copy(state)
        env.state = state
        u = np.random.uniform(-2, 2, size=(1,))
        before1, before2 = render(state, u)

        after_state = env.state #np.copy(state)
        after1, after2 = render(after_state, u)

        before = np.hstack((before1, before2))
        after = np.hstack((after1, after2))

        shard_no = i // (sample_size // num_shards)

        shard_path = path.join('{:03d}-of-{:03d}'.format(shard_no, num_shards))

        if not path.exists(path.join(output_dir, shard_path)):
            os.makedirs(path.join(output_dir, shard_path))

        before_file = path.join(shard_path, 'before-{:05d}.jpg'.format(i))
        plt.imsave(path.join(output_dir, before_file), before)

        after_file = path.join(shard_path, 'after-{:05d}.jpg'.format(i))
        plt.imsave(path.join(output_dir, after_file), after)

        samples.append({
            'before_state': initial_state.tolist(),
            'after_state': after_state.tolist(),
            'before': before_file,
            'after': after_file,
            'control': u.tolist(),
        })

    with open(path.join(output_dir, 'data.json'), 'wt') as outfile:
        json.dump(
            {
                'metadata': {
                    'num_samples': sample_size,
                    'step_size': step_size,
                    'apply_control': apply_control,
                    'time_created': str(datetime.now()),
                    'version': 1
                },
                'samples': samples
            }, outfile, indent=2)

    env.close()


def main(args):
    sample_size = args.sample_size

    sample_pendulum(sample_size=sample_size)


if __name__ == "__main__":
    # the default value is used for the planar task
    with open('../config.yaml') as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader)).sample

    main(config)