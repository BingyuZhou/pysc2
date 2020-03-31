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
        # upgrades
        self.embed_upgrads = keras.layers.Dense(64, activation=tf.tanh)
        # player (agent statistics)
        self.embed_player = keras.layers.Dense(64, activation=tf.nn.relu)
        # available_actions
        self.embed_available_act = keras.layers.Dense(64,
                                                      activation=tf.nn.relu)
        # race_requested
        self.embed_race = keras.layers.Dense(64, activation=tf.nn.relu)
        #
        self.embed_available_act = keras.layers.Dense(64,
                                                      activation=tf.nn.relu)
        self.embed_available_act = keras.layers.Dense(64,
                                                      activation=tf.nn.relu)
        self.embed_available_act = keras.layers.Dense(64,
                                                      activation=tf.nn.relu)
        self.embed_available_act = keras.layers.Dense(64,
                                                      activation=tf.nn.relu)

        self.embed_minimap = keras.layers.Conv2D(32,
                                                 1,
                                                 padding='same',
                                                 activation=tf.nn.relu)
        self.embed_minimap_2 = keras.layers.Conv2D(64,
                                                   3,
                                                   padding='same',
                                                   activation=tf.nn.relu)
        self.embed_minimap_3 = keras.layers.Conv2D(128,
                                                   3,
                                                   padding='same',
                                                   activation=tf.nn.relu)
        """
        Output
        """
        #TODO: autoregressive embedding
        self.action_id_logits = keras.layers.Dense(NUM_ACTION_FUNCTIONS)
        self.delay_logits = keras.layers.Dense(128)
        self.queued_logits = keras.layers.Dense(2)
        self.selected_units_logits = keras.layers.Dense(64)
        self.target_unit_logits = keras.layers.Dense(32)
        self.target_location_logits = keras.layers.Conv2D(1, 1, padding='same')

    # action distribution
    def call(self, obs):
        """
        Embedding of inputs
        """
        """ 
        Scalar features
        
        These are embedding of scalar features
        """
        embed_player = self.embed_player(np.log(obs.player + 1))
        embed_race = self.embed_race(
            tf.one_hot([obs.home_race_requested, obs.away_race_requested], 5))
        #FIXME: boolen vector of upgrades
        embed_upgrades = self.embed_upgrads(obs.upgrades)

        available_act_bool_vec = np.zeros(NUM_ACTION_FUNCTIONS,
                                          dtype=np.float32)
        available_act_bool_vec[obs.available_actions] = 1
        embed_available_act = self.embed_available_act(available_act_bool_vec)

        scalar_out = tf.concat(
            [embed_player, embed_race, embed_upgrades, embed_available_act],
            axis=1)
        """ 
        Map features 
        
        These are embedding of map features
        """
        embed_minimap = self.embed_minimap(obs.feature_minimap)
        embed_minimap = self.embed_minimap_2(embed_minimap)
        embed_minimap = self.embed_minimap_3(embed_minimap)
        embed_screen = self.embed_minimap(obs.feature_screen)
        embed_screen = self.embed_minimap_2(embed_screen)
        embed_screen = self.embed_minimap_3(embed_screen)

        map_out = tf.concat([embed_minimap, embed_screen], axis=3)

        #TODO: entities feature
        """
        State representation
        """
        # core
        scalar_out_2d = tf.tile(
            tf.expand_dims(tf.expand_dims(scalar_out, 1), 2),
            [1, map_out.shape[1], map_out.shape[2], 1])
        core_out = tf.concat([scalar_out_2d, map_out], axis=3)
        core_out_flat = keras.backend.flatten(core_out)
        """
        Decision output
        """
        # value
        value_out = keras.layers.Dense(1)
        # action id
        action_id_out = self.action_id_logits(core_out_flat)
        action_id_out = tf.nn.softmax(action_id_out)
        # delay
        delay_out = self.delay_logits(core_out_flat)
        delay_out = tf.nn.softmax(delay_out)

        # queued
        queued_out = self.queued_logits(core_out_flat)
        queued_out = tf.nn.softmax(queued_out)
        # selected units
        selected_out = self.selected_units_logits(core_out_flat)
        selected_out = tf.nn.softmax(selected_out)
        # target unit
        target_unit_out = self.target_unit_logits(core_out_flat)
        target_unit_out = tf.nn.softmax(target_unit_out)
        # target location
        target_location_out = self.target_location_logits(core_out)
        target_location_out = keras.backend.flatten(target_location_out)
        target_location_out = tf.nn.softmax(target_location_out)

        out = {
            'value': value_out,
            'action_id': action_id_out,
            'delay': delay_out,
            'queued': queued_out,
            'selected': selected_out,
            'target_unit': target_unit_out,
            'target_location': target_location_out
        }

        return out

    # action sampling
    def step(self, obs):
        """Sample actions and compute logp(a|s)"""
        out = self.call(obs)

        available_act_mask = np.zeros(NUM_ACTION_FUNCTIONS, dtype=np.float32)
        available_act_mask[obs.available_actions] = 1
        out['action_id'] *= available_act_mask

        action_id = tf.random.categorical(out['action_id'], 1)

        #TODO: Fill out args based on sampled action type
        for arg_type in obs.


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
        batch_logp = []  # batch logp(a|s)
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
            batch_obs.append(obs)

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
