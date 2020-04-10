import numpy as np


class Buffer():
    """Replay buffer"""
    def __init__(self):
        # self.obs is intentionally not initialized here, since it is Dict
        self.batch_act_id = []  # batch action
        self.batch_act_args = []
        self.batch_ret = []  # batch return
        self.batch_len = []  # batch trajectory length
        self.batch_logp = []  # batch logp(a|s)
        self.ep_rew = []  # episode rewards (trajectory rewards)
        self.ep_len = 0  # length of trajectory

    def add(self, obs, act_id, act_args, logp_a, reward):
        """Add one entry"""
        if (self.ep_len == 0):
            # first entry, needs to initialize self.obs
            self.batch_obs = obs
        else:
            for o in obs:
                self.batch_obs[o].append(obs[o][0])
        self.batch_act_id.append(act_id)
        self.batch_act_args.append(act_args)
        self.batch_logp.append(logp_a)
        self.ep_len += 1
        self.ep_rew.append(reward)

    def finalize(self, reward):
        """Finalize one trajectory"""
        self.ep_rew.append(reward)
        ret = np.sum(self.ep_rew)
        self.batch_ret += [ret] * self.ep_len
        self.batch_len.append(self.ep_len)

        # reset
        self.ep_len = 0
        self.ep_rew.clear()

    def size(self):
        return len(self.batch_ret)

    def sample(self):
        """Return buffer elements"""
        return [
            self.batch_obs, self.batch_act_id, self.batch_act_args,
            self.batch_ret
        ]
