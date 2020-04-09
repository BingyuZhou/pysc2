# Boilerplate Code for DRL with Tensorboard
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from buffer import Buffer

import tensorflow as tf
from tensorflow import keras
import tensorboard as tb
import numpy as np
import datetime
from absl import app, flags

from pysc2.env import sc2_env
from pysc2.lib import actions, features
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
EPS = 1E-8

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


class GLU(keras.Model):
    """Gated linear unit"""
    def __init__(self, input_size, out_size):
        super(GLU, self).__init__(name='GLU')
        self.input_size = input_size
        self.out_size = out_size
        self.layer1 = keras.layers.Dense(input_size, activation='sigmoid')
        self.layer2 = keras.layers.Dense(out_size)

    def call(self, input, context):
        x = self.layer1(context)  # gate
        x = x * input  # gated input
        x = self.layer2(x)
        return x


class Actor_Critic(keras.Model):
    def __init__(self):
        super(Actor_Critic, self).__init__(name='ActorCritic')
        # upgrades
        self.embed_upgrads = keras.layers.Dense(64, activation='tanh')
        # player (agent statistics)
        self.embed_player = keras.layers.Dense(64, activation='relu')
        # available_actions
        self.embed_available_act = keras.layers.Dense(64, activation='relu')
        # race_requested
        self.embed_race = keras.layers.Dense(64, activation='relu')
        # minimap feature
        self.embed_minimap = keras.layers.Conv2D(32,
                                                 1,
                                                 padding='same',
                                                 activation="relu")
        self.embed_minimap_2 = keras.layers.Conv2D(64,
                                                   3,
                                                   padding='same',
                                                   activation='relu')
        self.embed_minimap_3 = keras.layers.Conv2D(128,
                                                   3,
                                                   padding='same',
                                                   activation='relu')
        # screen feature
        # self.embed_screen = keras.layers.Conv2D(32,
        #                                         1,
        #                                         padding='same',
        #                                         activation=tf.nn.relu)
        # self.embed_screen_2 = keras.layers.Conv2D(64,
        #                                           3,
        #                                           padding='same',
        #                                           activation=tf.nn.relu)
        # self.embed_screen_3 = keras.layers.Conv2D(128,
        #                                           3,
        #                                           padding='same',
        #                                           activation=tf.nn.relu)
        # core
        self.flat = keras.layers.Flatten()
        """
        Output
        """
        #TODO: autoregressive embedding
        self.action_id_layer = keras.layers.Dense(256)
        self.action_id_gate = GLU(input_size=256,
                                  out_size=NUM_ACTION_FUNCTIONS)
        self.delay_logits = keras.layers.Dense(128)
        self.queued_logits = keras.layers.Dense(2)
        self.select_point_logits = keras.layers.Dense(4)
        self.select_add_logits = keras.layers.Dense(2)
        # self.select_unit_act=keras.layers.Dense(4)
        # self.selec_unit_id_logits=keras.layers.Dense(64)
        self.select_worker_logits = keras.layers.Dense(4)
        self.target_unit_logits = keras.layers.Dense(32)
        self.target_location_flat = keras.layers.Flatten()
        self.target_location_logits = keras.layers.Conv2D(1, 1, padding='same')

        self.value = keras.layers.Dense(1)

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
            tf.one_hot(
                [obs.home_race_requested[0], obs.away_race_requested[0]],
                depth=4))
        #FIXME: boolen vector of upgrades, size is unknown
        upgrades_bool_vec = np.zeros((1, 20), dtype=np.float32)
        upgrades_bool_vec[obs.upgrades] = 1.
        embed_upgrades = self.embed_upgrads(upgrades_bool_vec, )

        available_act_bool_vec = np.zeros((1, NUM_ACTION_FUNCTIONS),
                                          dtype=np.float32)
        available_act_bool_vec[obs.available_actions] = 1
        embed_available_act = self.embed_available_act(available_act_bool_vec)

        scalar_out = tf.concat([
            embed_player,
            tf.reshape(embed_race, [1, -1]), embed_upgrades,
            embed_available_act
        ],
                               axis=1)
        # print("scalar_out: {}".format(scalar_out.shape))
        """ 
        Map features 
        
        These are embedding of map features
        """
        def one_hot_map(obs, screen_on=False):
            if screen_on:
                Features = features.SCREEN_FEATURES
            else:
                Features = features.MINIMAP_FEATURES
            out = []
            for feature in Features:
                if feature.type is features.FeatureType.CATEGORICAL:
                    one_hot = tf.one_hot(obs[feature.name],
                                         depth=feature.scale)
                else:  # features.FeatureType.SCALAR
                    one_hot = obs[feature.name] / 255.0

                out.append(one_hot)
            out = tf.concat(out, axis=-1)
            return out

        one_hot_minimap = one_hot_map(obs.feature_minimap)
        embed_minimap = self.embed_minimap(one_hot_minimap)
        # embed_minimap = self.embed_minimap_2(embed_minimap)
        # embed_minimap = self.embed_minimap_3(embed_minimap)

        # one_hot_screen = one_hot_map(obs.feature_screen, screen_on=True)
        # embed_screen = self.embed_screen(one_hot_screen)
        # embed_screen = self.embed_screen_2(embed_screen)
        # embed_screen = self.embed_screen_3(embed_screen)
        # map_out = tf.concat([embed_minimap, embed_screen], axis=-1)
        map_out = embed_minimap
        # print("map_out: {}".format(map_out.shape))

        #TODO: entities feature
        """
        State representation
        """
        # core
        scalar_out_2d = tf.tile(
            tf.expand_dims(tf.expand_dims(scalar_out, 1), 2),
            [1, map_out.shape[1], map_out.shape[2], 1])
        core_out = tf.concat([scalar_out_2d, map_out], axis=3)
        core_out_flat = self.flat(core_out)
        """
        Decision output
        """
        # value
        value_out = self.value(core_out_flat)
        # action id
        action_id_out = self.action_id_layer(core_out_flat)
        action_id_out = self.action_id_gate(action_id_out, embed_available_act)
        # delay
        delay_out = self.delay_logits(core_out_flat)

        # queued
        queued_out = self.queued_logits(core_out_flat)
        # selected units
        select_point_out = self.select_point_logits(core_out_flat)

        select_add_out = self.select_add_logits(core_out_flat)

        select_worker_out = self.select_worker_logits(core_out_flat)
        # target unit
        target_unit_out = self.target_unit_logits(core_out_flat)
        # target location
        target_location_out = self.target_location_logits(core_out)
        _, self.location_out_width, self.location_out_height, _ = target_location_out.shape

        target_location_out = self.target_location_flat(target_location_out)

        out = {
            'value': value_out,
            'action_id': action_id_out,
            'delay': delay_out,
            'queued': queued_out,
            'select_point_act': select_point_out,
            'select_add': select_add_out,
            'select_worker': select_worker_out,
            'target_unit': target_unit_out,
            'target_location': target_location_out
        }

        return out

    # action sampling
    def step(self, obs, action_spec):
        """Sample actions and compute logp(a|s)"""
        out = self.call(obs)

        available_act_mask = np.ones(NUM_ACTION_FUNCTIONS,
                                     dtype=np.float32) * EPS
        available_act_mask[obs.available_actions] = 1.0
        # HACK: intentionally turn off control group, select_unit !!
        available_act_mask[[4, 5]] = EPS
        out['action_id'] = tf.math.softmax(
            out['action_id']) * available_act_mask
        # renormalize
        out['action_id'] /= tf.reduce_sum(out['action_id'],
                                          axis=-1,
                                          keepdims=True)
        out['action_id'] = tf.math.log(out['action_id'])

        action_id = tf.random.categorical(out['action_id'], 1).numpy().item()

        # Fill out args based on sampled action type
        args_out = []
        logp_a = tf.reduce_sum(
            out['action_id'] *
            tf.one_hot(action_id, depth=NUM_ACTION_FUNCTIONS),
            axis=-1)

        for arg_type in action_spec.functions[action_id].args:
            if arg_type.name in ['screen', 'screen2', 'minimap']:
                location_id = tf.random.categorical(out['target_location'],
                                                    1).numpy().item()
                x, y = location_id % self.location_out_width, location_id // self.location_out_width
                args_out.append([x, y])

                logp_a += tf.reduce_sum(out['target_location'] * tf.one_hot(
                    location_id,
                    depth=self.location_out_width * self.location_out_height),
                                        axis=-1)
            else:
                # non-spatial args
                sample = tf.random.categorical(out[arg_type.name],
                                               1).numpy().item()
                args_out.append([sample])
                logp_a += tf.reduce_sum(
                    out[arg_type.name] *
                    tf.one_hot(sample, depth=arg_type.sizes[0]),
                    axis=-1)

        return out['value'].numpy().item(), action_id, args_out, logp_a.numpy(
        ).item()

    def logp_a(self, action_id, action_args, pi, action_spec):
        """logp(a|s)"""
        logp = tf.reduce_sum(pi['action_id'] *
                             tf.one_hot(action_id, depth=NUM_ACTION_FUNCTIONS),
                             axis=-1)
        # args
        for ind, arg_type in enumerate(action_spec.functions[action_id].args):
            if arg_type.name in ['screen', 'screen2', 'minimap']:
                location_id = action_args[ind][
                    1] * self.location_out_width + action_args[ind][0]
                logp += tf.reduce_sum(pi['target_location'] * tf.one_hot(
                    location_id,
                    depth=self.location_out_width * self.location_out_height),
                                      axis=-1)
            else:
                # non-spatial args
                logp += tf.reduce_sum(
                    pi[arg_type.name] *
                    tf.one_hot(action_args[ind], depth=arg_type.sizes[0]),
                    axis=-1)

        return logp

    def loss(self, obs, act_id, act_args, ret, action_spec):
        # expection grad log
        out = self.call(obs)

        logp = self.logp_a(act_id, act_args, out, action_spec)

        return -tf.reduce_mean(logp * ret)


def expand_dim(obs):
    """Expand dimention of observation to meet requirements of NN"""
    for o in obs:
        obs[o] = np.expand_dims(obs[o], axis=0)
    return obs


# run one policy update
def train(env_name, batch_size, epochs):
    actor_critic = Actor_Critic()

    optimizer = keras.optimizers.Adam()
    # set env
    with SC2EnvWrapper(
            map_name=env_name,
            players=[sc2_env.Agent(sc2_env.Race.random)],
            agent_interface_format=sc2_env.parse_agent_interface_format(
                feature_minimap=32, feature_screen=1),
            step_mul=FLAGS.step_mul,
            game_steps_per_episode=FLAGS.game_steps_per_episode,
            disable_fog=FLAGS.disable_fog) as env:

        action_spec = env.action_spec()[0]  # assume one agent

        def train_one_epoch():
            # initialize replay buffer
            buffer = Buffer()

            # initial observation
            timeStepTuple = env.reset()
            step_type, reward, discount, obs = timeStepTuple[0]
            obs = expand_dim(obs)
            # render first episode of each epoch
            render_env = True

            # fill in recorded trajectories
            while True:
                env.render(render_env)

                # print("computing action ...")
                v, act_id, act_args, logp_a = actor_critic.step(
                    obs, action_spec)

                # print("buffer logging ...")
                buffer.add(obs, act_id, act_args, logp_a, reward)
                # print("apply action in env ...")
                timeStepTuple = env.step(
                    [actions.FunctionCall(act_id, act_args)])
                step_type, reward, discount, obs = timeStepTuple[0]
                obs = expand_dim(obs)

                if step_type == step_type.LAST:
                    buffer.finalize(reward)

                    # respawn env
                    _, _, _, obs = env.reset()[0]

                    # stop render
                    render_env = True

                    if buffer.size() > batch_size:
                        break

            def train_step(buffer_sample):
                obs, act_id, act_args, ret = buffer_sample
                with tf.GradientTape() as tape:
                    ls = actor_critic.loss(obs, act_id, act_args, ret,
                                           action_spec)
                grad = tape.gradient(ls, actor_critic.trainable_variables)
                optimizer.apply_gradients(
                    zip(grad, actor_critic.trainable_variables))
                return ls

            # update policy
            batch_loss = train_step(buffer.sample())

            return batch_loss, batch_ret, batch_len

        for i in range(epochs):
            batch_loss, batch_ret, batch_len = train_one_epoch()
            with train_summary_writer.as_default():
                tf.summary.scalar('batch_ret', np.mean(batch_ret), step=i)
                tf.summary.scalar('batch_len', np.mean(batch_len), step=i)

            print(
                "epoch {0:2d} loss {1:.3f} batch_ret {2:.3f} batch_len {3:.3f}"
                .format(i, batch_loss.numpy(), np.mean(batch_ret),
                        np.mean(batch_len)))


def main(argv):
    epochs = 500
    batch_size = 100
    train(FLAGS.env_name, batch_size, epochs)


if __name__ == '__main__':
    app.run(main)
