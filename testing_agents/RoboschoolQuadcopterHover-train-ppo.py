#!/usr/bin/env python

import argparse
import time
import os
import sys
import json

from baselines import bench
from baselines import logger
from baselines.common import tf_util as U
from baselines.common.misc_util import set_global_seeds, boolean_flag, SimpleMonitor
from baselines.common.mpi_fork import mpi_fork
from baselines.logger import Logger

import gym, logging
from mpi4py import MPI

import tensorflow as tf
import roboschool

eval_env = None
log_dir = None

def callback(l, g):
    """ Visualize a sample flight every few trials """

    #return

    global log_dir
    idx = l['iters_so_far']
    if log_dir is not None and (idx % 50) == 0:
        from baselines.common.tf_util import save_state
        import os
        fn = os.path.join(log_dir, "model{}.cpkt".format(idx))
        save_state(fn)


    global eval_env
    if eval_env is None:
        return

    video = True

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

def every_hundred(idx):
    return ((idx+1) % 100) == 0

def train(env_id, num_timesteps, seed, num_cpu, logdir, gym_monitor=False, evaluation=False, bind_to_core=False, **kwargs):

    #assert not (gym_monitor and evaluation), "Unfortunately trying to monitor both training and testing with video writing causes a segfault"

    # currently doesn't work unless environment works natively from bash
    #kwargs['logdir'] = logdir
    #whoami = mpi_fork(num_cpu, bind_to_core=bind_to_core)
    #if whoami == 'parent':
    #    sys.exit(0)

    ## Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    #if rank != 0:
    #    # Write to temp directory for all non-master workers.
    #    actual_dir = None
    #    Logger.CURRENT.close()
    #    Logger.CURRENT = Logger(dir=mkdtemp(), output_formats=[])
    #    logger.set_level(logger.DISABLED)
    #else:
    #    logger.session(dir=logdir).__enter__()

    # store this so callback can write model to disk
    global log_dir
    log_dir = logdir

    from baselines.pposgd import mlp_policy, pposgd_simple
    U.make_session(num_cpu=num_cpu).__enter__()
    logger.session(dir=logdir).__enter__()

    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        from MlpPolicy import MlpPolicy
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, nonlinear_function=tf.nn.relu)
        #return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        #    hid_size=64, num_hid_layers=3)

    # Create envs.
    if gym_monitor and logdir:
        print("Logger dir: {}".format(logger.get_dir()))
        env = gym.wrappers.Monitor(env, os.path.join(logdir, 'gym_train'), force=True, video_callable=is_perfect_cube)
    else:
        env = bench.Monitor(env, os.path.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)

    if evaluation  and logdir:
        global eval_env
        eval_env = gym.make(env_id)
        #eval_env = gym.wrappers.Monitor(eval_env, os.path.join(logdir, 'gym_eval'), force=True, video_callable=is_perfect_cube)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Train model.
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=500,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64*4,
            gamma=0.99, lam=0.95,
            schedule='linear',
            callback=callback
        )
    env.close()
    if eval_env is not None:
        eval_env.close()
    Logger.CURRENT.close()

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--env-id', type=str, default='RoboschoolQuadcopterHover-v0')
    parser.add_argument('--num-cpu', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=1e8)
    parser.add_argument('--logdir', type=str, default=None)
    boolean_flag(parser, 'gym-monitor', default=True)
    boolean_flag(parser, 'evaluation', default=False)
    boolean_flag(parser, 'bind-to-core', default=False)

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




