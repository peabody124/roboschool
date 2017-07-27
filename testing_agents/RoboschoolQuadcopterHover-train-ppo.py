#!/usr/bin/env python

import argparse
import time
import os
import sys
import json

from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import gym, logging
from baselines import logger
import roboschool
from baselines.common.misc_util import boolean_flag, SimpleMonitor

eval_env = None

def callback(l, g):
    """ Visualize a sample flight every few trials """

    global eval_env
    if eval_env is None:
        return

    video = False

    pi = l['pi']

    # visualize a flight from this policy
    t = 0
    ac = eval_env.action_space.sample() # not used, just so we have the datatype
    ob = eval_env.reset()

    done = False

    while not done:
        ac = pi.act(False, ob)
        ac = ac[0]

        ob, rew, done, _ = eval_env.step(ac)

        if video:
            eval_env.render("human")

def is_perfect_cube(idx):
    return round(idx ** (1./3)) ** 3.0 == idx

def train(env_id, num_timesteps, seed, num_cpu, logdir, gym_monitor=False, evaluation=False):

    assert not (gym_monitor and evaluation), "Unfortunately trying to monitor both training and testing with video writing causes a segfault"

    from baselines.pposgd import mlp_policy, pposgd_simple
    U.make_session(num_cpu=num_cpu).__enter__()
    logger.session(dir=logdir).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    if gym_monitor and logdir:
        print("Logger dir: {}".format(logger.get_dir()))
        env = gym.wrappers.Monitor(env, os.path.join(logdir, 'gym_train'), force=True, video_callable=is_perfect_cube)
    else:
        env = bench.Monitor(env, os.path.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)

    if evaluation  and logdir:
        global eval_env
        eval_env = gym.make(env_id)
        eval_env = gym.wrappers.Monitor(eval_env, os.path.join(logdir, 'gym_eval'), force=True, video_callable=is_perfect_cube)

    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95,
            callback=callback
        )
    env.close()

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--env-id', type=str, default='RoboschoolQuadcopterHover-v0')
    parser.add_argument('--num-cpu', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=1e6)
    parser.add_argument('--logdir', type=str, default=None)
    boolean_flag(parser, 'gym-monitor', default=True)
    boolean_flag(parser, 'evaluation', default=False)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()

    # Figure out what logdir to use.
    if args['logdir'] is None:
        args['logdir'] = os.getenv('OPENAI_LOGDIR')
    
    # Print and save arguments.
    logger.info('Arguments:')
    for key in sorted(args.keys()):
        logger.info('{}: {}'.format(key, args[key]))
    logger.info('')
    if args['logdir']:
        with open(os.path.join(args['logdir'], 'args.json'), 'w') as f:
            json.dump(args, f)

    train(**args)
