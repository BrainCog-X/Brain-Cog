# Code Reference:
# https://github.com/deepmind/deepmind-research/blob/master/side_effects_penalties/side_effects_penalty.py
# https://github.com/alexander-turner/attainable-utility-preservation/blob/master/agents/model_free_aup.py


import numpy as np
from collections import defaultdict


class StepwiseInactionModel(object):
    """Calculate the next state after one noop action from current state"""

    def __init__(self, noop_action=None):
        self._noop_action = noop_action
        self._baseline_state = None
        self._inaction_model = defaultdict(lambda: defaultdict(lambda: 0))  # init _inaction_model[state][next_state]=0
        return

    def reset(self, baseline_state):
        self._baseline_state = baseline_state
        return

    def _sample(self, state):
        """Sample next_state based on its history frequency"""
        d = self._inaction_model[state]
        counts = np.array(list(d.values()))
        assert len(counts) > 0 and sum(counts) > 0
        index = np.random.choice(a=len(counts), p=counts / sum(counts))
        return list(d.keys())[index]

    def calculate(self, prev_state, prev_action, current_state):
        """Update inaction transition model, and predict the noop baseline state """
        # update
        if prev_action == self._noop_action:
            self._inaction_model[prev_state][current_state] += 1
        # predict
        if prev_state in self._inaction_model:
            self._baseline_state = self._sample(prev_state)
        else:
            self._baseline_state = prev_state
        return self._baseline_state


class AttainableUtilityMeasure(object):
    def __init__(self, uf_num=10, uf_discount=0.99):

        # initialize a group of auxiliary utility functions
        self._uf_values = [defaultdict(lambda: 0.0) for _ in range(uf_num)]
        # initialize random rewards for auxiliary tasks
        self._uf_rewards = [defaultdict(lambda: np.random.random()) for _ in range(uf_num)]

        assert 0 <= uf_discount < 1.0, "uf_discount should be between [0, 1)"
        self._uf_discount = uf_discount

        # initialize update counts and confidence for the value estimation of each state
        self._uf_update_cnts = [defaultdict(lambda: 0) for _ in range(uf_num)]
        self._confid_func = lambda x: 1.0 if x > 0 else 0.0  # confident if state value has been updated

        # record predecessors of states for backward value iteration 
        self._predecessors = defaultdict(set)
        return

    def update(self, prev_state, prev_action, current_state):
        """Update estimations of Auxiliary Utility Functions with new transitions"""
        del prev_action  # unused in value iteration
        # update transitions
        self._predecessors[current_state].add(prev_state)
        # iterative update values
        for reward, u_value, update_cnt in zip(self._uf_rewards, self._uf_values, self._uf_update_cnts):
            seen = set()
            queue = [current_state]
            while queue:
                s_to = queue.pop(0)
                seen.add(s_to)
                for s_from in self._predecessors[s_to]:
                    v = reward[s_from] + self._uf_discount * u_value[s_to]
                    if u_value[s_from] < v:
                        u_value[s_from] = v
                        if s_from not in seen:
                            queue.append(s_from)
                    update_cnt[s_from] += 1  # update counts for the value estimation of each state
        return

    def calculate(self, current_state, baseline_state, dev_fun=lambda diff: abs(np.minimum(0, diff))):
        """Calculate the deviation between two states, with given deviation_function"""
        cs_values = [u_value[current_state] for u_value in self._uf_values]
        bs_values = [u_value[baseline_state] for u_value in self._uf_values]
        diff_values = [(cs_value - bs_value) for cs_value, bs_value in zip(cs_values, bs_values)]

        cs_confids = [self._confid_func(update_cnt[current_state]) for update_cnt in self._uf_update_cnts]
        bs_confids = [self._confid_func(update_cnt[baseline_state]) for update_cnt in self._uf_update_cnts]
        diff_confids = [(cs_confid * bs_confid) for cs_confid, bs_confid in zip(cs_confids, bs_confids)]

        deviations = [diff_confid * dev_fun(diff_value) * (1. - self._uf_discount)
                      for diff_confid, diff_value in zip(diff_confids, diff_values)]
        return sum(deviations) / len(deviations)

    def _get_aup_value(self, state):
        """For debugging purpose, 
        The Attainable Utility Preservation (aup) value are based on the estimation 
        towards an imaginary baseline_state of u_value=0.0 and confidence=1.0"""
        dev_fun = lambda diff: abs(diff)

        cs_values = [u_value[state] for u_value in self._uf_values]
        bs_values = [0.0] * len(self._uf_values)
        diff_values = [(cs_value - bs_value) for cs_value, bs_value in zip(cs_values, bs_values)]

        cs_confids = [self._confid_func(update_cnt[state]) for update_cnt in self._uf_update_cnts]
        bs_confids = [1.0] * len(self._uf_update_cnts)
        diff_confids = [(cs_confid * bs_confid) for cs_confid, bs_confid in zip(cs_confids, bs_confids)]

        deviations = [diff_confid * dev_fun(diff_value) * (1. - self._uf_discount)
                      for diff_confid, diff_value in zip(diff_confids, diff_values)]
        return sum(deviations) / len(deviations)

    def _get_avgd_confid(self, state):
        """For debugging purpose"""
        s_confids = [self._confid_func(update_cnt[state]) for update_cnt in self._uf_update_cnts]
        return sum(s_confids) / len(s_confids)

    def _get_u_values(self, state):
        """For debugging purpose"""
        return [u_value[state] for u_value in self._uf_values]
