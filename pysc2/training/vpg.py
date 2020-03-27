# Boilerplate Code for DRL with Tensorboard
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import tensorboard as tb
import numpy as np
import datetime
from absl import app, flags

from pysc2.env import sc2_env
from pysc2.lib import actions
from sc2env_wrapper import SC2EnvWrapper

tf.keras.backend.set_floatx('float32')
""" log info"""
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + current_time + '/train'
test_log_dir = 'logs/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
""" constants"""
NUM_ACTION_FUNCTIONS = 573

# args
FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "CollectMineralShards",
                    "Select which map to play.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_enum(
    "difficulty",
    "very_easy",
    sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
    "If agent2 is a built-in Bot, it's strength.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")


class Actor_Critic(keras.Model):
    def __init__(self):
        super(Actor_Critic, self).__init__()

        pass

    # action distribution
    def call(self, obs):
        x = self.layer1(obs)
        x = self.layer2(x)
        x = self.layer3(x)
        logp = tf.nn.log_softmax(x)
        return logp

    # action sampling
    def sample(self, obs):
        return tf.squeeze(tf.random.categorical(self.call(obs), 1), axis=1)

    def loss(self, obs, act, ret):
        # expection grad log
        mask = tf.one_hot(act, depth=self.act_size)
        logp_a = tf.reduce_sum(self.call(obs) * mask, axis=1)  # logp(a|s)
        loss = -tf.reduce_mean(logp_a * ret)
        return loss


# run one policy update
def train(env_name, batch_size, epochs):
    actor_critic = Actor_Critic()

    optimizer = keras.optimizers.Adam()
    # set env
    env = SC2EnvWrapper(
        map_name=env_name,
        players=[sc2_env.Agent(sc2_env.Race.random)],
        agent_interface_format=sc2_env.parse_agent_interface_format(
            feature_screen=64, feature_minimap=64),
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=FLAGS.game_steps_per_episode,
        disable_fog=FLAGS.disable_fog)

    def train_one_epoch():
        # initialize replay buffer
        batch_obs = []
        batch_act = []  # batch action
        batch_ret = []  # batch return
        batch_len = []  # batch trajectory length
        ep_rew = []  # episode rewards (trajectory rewards)
        ep_len = 0  # length of trajectory

        # initial observation
        timeStepTuple = env.reset()
        step_type, reward, discount, obs = timeStepTuple[0]

        # render first episode of each epoch
        render_env = True

        # fill in recorded trajectories
        while True:
            env.render(render_env)

            act = actor_critic.sample(obs)

            batch_act.append(act)
            batch_obs.append(obs.copy())

            timeStepTuple = env.step(act)
            step_type, reward, discount, obs = timeStepTuple[0]

            ep_rew.append(reward)
            ep_len += 1

            if step_type == step_type.LAST:
                # compute return
                ret = np.sum(ep_rew)
                batch_ret += [ret] * ep_len
                batch_len.append(ep_len)

                # respawn env
                obs = env.reset()
                ep_len = 0
                ep_rew.clear()

                # stop render
                render_env = False

                if len(batch_obs) > batch_size:
                    break

        @tf.function
        def train_step(obs, act, ret):
            with tf.GradientTape() as tape:
                ls = actor_critic.loss(obs, act, ret)
            grad = tape.gradient(ls, actor_critic.trainable_variables)
            optimizer.apply_gradients(
                zip(grad, actor_critic.trainable_variables))
            return ls

        # update policy
        batch_loss = train_step(tf.constant(batch_obs), np.array(batch_act),
                                tf.constant(batch_ret, dtype=tf.float32))

        return batch_loss, batch_ret, batch_len

    for i in range(epochs):
        batch_loss, batch_ret, batch_len = train_one_epoch()
        with train_summary_writer.as_default():
            tf.summary.scalar('batch_ret', np.mean(batch_ret), step=i)
            tf.summary.scalar('batch_len', np.mean(batch_len), step=i)

        print("epoch {0:2d} loss {1:.3f} batch_ret {2:.3f} batch_len {3:.3f}".
              format(i, batch_loss.numpy(), np.mean(batch_ret),
                     np.mean(batch_len)))


def main(argv):
    epochs = 500
    batch_size = 1000
    train(FLAGS.env_name, batch_size, epochs)


if __name__ == '__main__':
    app.run(main)
