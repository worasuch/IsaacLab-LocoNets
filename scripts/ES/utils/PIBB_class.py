# *******************************************
# *                                         *
# *                 PIBB                    *
# *                                         *
# *******************************************
#  created by: Worasuchad Haomachai
#  contract: haomachai@gmail.com
#  update: 25/03/2023
#  version: 0.0.0


# *******************************************
# *                                         *
# *               description               *
# *                                         *
# *******************************************
# adopted from:  M. Thor.


import math
import numpy as np
from operator import add


class PIBB(object):
    def __init__(self, _num_params,                 # number of model parameters 
                 _rollouts=512,                     # popsize 
                 _lambda=10,                        # exploration constant
                 _lambda_decay=1,                   # exploration decay
                 _sigma_init=0.045,                 # variance noise
                 _sigma_decay=0.995):               # varionce noise decay   
        
        self.num_params = _num_params
        self.rollouts   = _rollouts
        self.h          = _lambda
        self.decay      = _lambda_decay
        self.sigma      = _sigma_init
        self.sigma_decay = _sigma_decay

        self.reward = np.zeros(self.rollouts)
        self.mu     = np.zeros(self.num_params)
        self.best_mu = np.zeros(self.num_params)
        self.best_fitness = -np.inf


    def set_mu(self, mu):
        self.mu = np.array(mu)


    def ask(self):
        # normal dist noise
        self.epsilon = np.random.randn(self.rollouts, self.num_params) * self.sigma

        self.solutions = self.mu.reshape(1, self.num_params) + self.epsilon
        return self.solutions


    def self_devision(self, n, d):
        return n/d if d else 0
    

    def tell(self, _reward_table_result):
        # input must be a numpy float array
        assert (len(_reward_table_result) == self.rollouts), "Inconsistent reward_table size reported."
        self.reward = np.array(_reward_table_result)

        # calculate fitness min, max
        max_fitness = np.max(self.reward)
        min_fitness = np.min(self.reward)

        if max_fitness > self.best_fitness:
            self.best_fitness = max_fitness
            self.best_mu = self.solutions[np.argmax(self.reward)]

        # run algorithm PIBB
        s_norm  = np.zeros(self.rollouts)
        p       = np.zeros(self.rollouts)

        # compute trajectory cost/fitness
        for k in range(self.rollouts):
            s_norm[k]   = np.exp(self.h * self.self_devision((self.reward[k] - min_fitness), (max_fitness - min_fitness)))

        # compute probability for each rollout
        for k in range(self.rollouts):
            p[k] = s_norm[k] / np.sum(s_norm)

            # cost-weighted averaging
            self.epsilon[k] = [x*p[k] for x in self.epsilon[k]]

            # update policy parameters
            self.mu = np.asarray(list(map(add, self.mu, self.epsilon[k])))
    

        # decay h (1/λ) / variance as learning progress
        self.h = self.decay * (1/self.h)
        self.h = (1/self.h)

        # sigma decay
        self.sigma *= self.sigma_decay

    def best_param(self):
        return self.best_mu