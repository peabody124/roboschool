import os.path, time, gym
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import roboschool


def demo_run():
    config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        device_count = { "GPU": 0 } )
    sess = tf.InteractiveSession(config=config)

    env = gym.make("RoboschoolQuadcopterHover-v0")

    while 1:
        frame = 0
        score = 0
        restart_delay = 0.0
        obs = env.reset()

        a = np.zeros(4,)
        a[3] = 0.5

        while 1:

            obs, r, done, _ = env.step(a)

            score += r
            frame += 1
            still_open = env.render("human")
            if still_open==False:
                return
            if not done: continue
            if restart_delay==0.0:
                print("score=%0.2f in %i frames" % (score, frame))
                if still_open!=True:      # not True in multiplayer or non-Roboschool environment
                    break
                restart_delay = time.time() + 1.0
            if time.time() > restart_delay:
                break

if __name__=="__main__":
    demo_run()
