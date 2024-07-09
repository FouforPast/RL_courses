# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""DQN Trainer"""
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.api import ms_function
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
from mindspore_rl.agent.trainer import Trainer
from mindspore_rl.agent import trainer
import mindspore.numpy as np


class DQNTrainer(Trainer):
    """DQN Trainer"""

    def __init__(self, msrl, params):
        nn.Cell.__init__(self, auto_prefix=False)
        self.zero = Tensor(0, ms.float32)
        self.squeeze = P.Squeeze()
        self.less = P.Less()
        self.zero_value = Tensor(0, ms.float32)
        self.fill_value = Tensor(params['buffer_num_before_learning_begin'], ms.float32)
        self.inited = Parameter(Tensor((False,), ms.bool_), name='init_flag')
        self.mod = P.Mod()
        self.false = Tensor((False,), ms.bool_)
        self.true = Tensor((True,), ms.bool_)
        self.num_evaluate_episode = params['num_evaluate_episode']
        self.steps = Parameter(Tensor([1], ms.int32))
        self.update_period = Tensor(params['update_target_iter'], ms.float32)
        self.print = P.Print()
        self.argmax = P.Argmax()
        self.mul = P.Mul()
        self.div = P.Div()
        self.cnt = ms.Parameter(Tensor([0], ms.float32))
        self.thre = ms.Parameter(Tensor([0.5], ms.float32))
        self.java2python_idx = ms.Tensor(np.array([25, -1, 23, 21, 15, -1, 24, 22, 14, -1, 37, 20, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 38, 39, -1, -1, 26, 27,
            16, 17, 0, 1, 6, 7, 35, 36, 18, 19, 10, 11, 12, 13, 30, 31, 33,
            34, 28, 29, 4, 5, 8, 9, 2, 3, 32]), ms.int32)
        self.max_eposide = ms.Parameter(Tensor([1000], ms.float32))
        # self.log0 = open('log0.txt', 'w')
        self.obs = ms.Parameter(ms.Tensor(np.zeros((144,)), ms.float32))
        self.randreal = P.UniformReal()
        super(DQNTrainer, self).__init__(msrl)

    def trainable_variables(self):
        """Trainable variables for saving."""
        trainable_variables = {"policy_net": self.msrl.learner.policy_network}
        return trainable_variables
    
    @ms_function
    def getState(self):
        # self.print(self.obs)
        s1 = ms.Tensor(np.array([self.obs[0], self.obs[1], self.obs[2], self.obs[3], self.obs[4], self.obs[5], self.obs[6], self.obs[7], self.obs[8], self.obs[65],  # 己方人物状态
                        self.obs[66], self.obs[67], self.obs[68], self.obs[69], self.obs[70], self.obs[71], self.obs[72], self.obs[73], self.obs[74], self.obs[131], # 敌方人物状态
                        self.obs[132], self.obs[133], self.obs[134], self.obs[135], self.obs[136], self.obs[137], # 已方抛射物状态
                        self.obs[138], self.obs[139], self.obs[140], self.obs[141], self.obs[142], self.obs[143]]), ms.float32) # 敌方抛射物状态
        s2 = ms.Tensor(np.array([self.obs[66], self.obs[67], self.obs[68], self.obs[69], self.obs[70], self.obs[71], self.obs[72], self.obs[73], self.obs[74], self.obs[131],  # 己方人物状态
                        self.obs[0], self.obs[1], self.obs[2], self.obs[3], self.obs[4], self.obs[5], self.obs[6], self.obs[7], self.obs[8], self.obs[65], # 敌方人物状态
                        self.obs[138], self.obs[139], self.obs[140], self.obs[141], self.obs[142], self.obs[143], # 已方抛射物状态
                        self.obs[132], self.obs[133], self.obs[134], self.obs[135], self.obs[136], self.obs[137]]), ms.float32)
        a1 = ms.Tensor(np.array([self.java2python_idx[self.argmax(self.obs[9:65])]]), ms.int32)
        a2 = ms.Tensor(np.array([self.java2python_idx[self.argmax(self.obs[75:131])]]), ms.int32)
        return s1, s2, a1, a2
        # s1, s2, a1, a2 = None, None, None, None
        # return s1, s2, a1, a2

    @ms_function
    def init_training(self):
        """Initialize training"""
        state = self.msrl.collect_environment.reset()
        # self.print(state)
        done = self.false
        i = self.zero_value
        self.obs = state
        s1, s2, a1, a2 = self.getState()
        while self.less(i, self.fill_value):
            done, _, new_state, action, my_reward = self.msrl.agent_act(
                trainer.INIT, s1)
            # self.msrl.replay_buffer_insert(
            #     [state, action, my_reward, new_state])
            # state = new_state
            # if done:
            #     state = self.msrl.collect_environment.reset()
            #     done = self.false
            if not done:
                self.obs = new_state
                s12, s22, a12, a22 = self.getState()
                if a1 != -1:
                    self.msrl.replay_buffer_insert(
                        [s1, a1, my_reward, s12])
                if a2 != -1:
                    self.msrl.replay_buffer_insert(
                        [s2, a2, -my_reward, s22])
                # self.msrl.replay_buffer_insert(
                #     [state, action, my_reward, new_state])
                state = new_state
                s1, s2, a1, a2 = s12, s22, a12, a22
            else:
                state = self.msrl.collect_environment.reset()
                self.obs = state
                s1, s2, a1, a2 = self.getState()
                done = self.false
            i += 1
        return done

    @ms_function
    def train_one_episode(self):
        """Train one episode"""
        if not self.inited:
            self.init_training()
            self.inited = self.true
        self.cnt += 1
        # self.thre = ms.Parameter(Tensor([self.cnt / self.max_eposide * 4 / 3], ms.float32))
        self.thre = self.cnt / self.max_eposide * 4 / 3
        self.print(self.thre)

        state = self.msrl.collect_environment.reset()
        self.obs = state
        s1, s2, a1, a2 = self.getState()
        done = self.false
        total_reward = self.zero
        steps = self.zero
        loss = self.zero
        while not done:
            done, r, new_state, action, my_reward = self.msrl.agent_act(
                trainer.COLLECT, s1)
            if done:
                break
            self.obs = new_state
            s12, s22, a12, a22 = self.getState()
            if a1 != -1:
                self.msrl.replay_buffer_insert(
                    [s1, a1, my_reward, s12])
            if a2 != -1 and self.randreal((1,)) > self.thre:
                self.msrl.replay_buffer_insert(
                    [s2, a2, -my_reward, s22])
            # self.msrl.replay_buffer_insert(
            #     [state, action, my_reward, new_state])
            state = new_state
            s1, s2, a1, a2 = s12, s22, a12, a22
            r = self.squeeze(r)
            loss = self.msrl.agent_learn(self.msrl.replay_buffer_sample())
            total_reward += r
            self.steps += 1
            steps += 1
            if not self.mod(self.steps, self.update_period):
                self.msrl.learner.update()
        # average_reward = total_reward / steps
        # self.log0.write(total_reward)
        
        return loss, total_reward, steps

    @ms_function
    def evaluate(self):
        """Policy evaluate"""
        total_reward = self.zero_value
        eval_iter = self.zero_value
        while self.less(eval_iter, self.num_evaluate_episode):
            episode_reward = self.zero_value
            state = self.msrl.eval_environment.reset()
            done = self.false
            self.print(eval_iter)
            while not done:
                self.obs = state
                s1, s2, a1, a2 = self.getState()
                done, r, state = self.msrl.agent_act(trainer.EVAL, s1)
                r = self.squeeze(r)
                episode_reward += r
            total_reward += episode_reward
            eval_iter += 1
        avg_reward = total_reward / self.num_evaluate_episode
        return avg_reward
    
# if __name__ == '__main__':
#     q = DQNTrainer()
#     q.train()
