import random
import numpy as np
import tensorflow as tf

class SpecAugment():
    def __init__(self, policy, zero_mean_normalized=True):
        self.policy = policy
        self.zero_mean_normalized = zero_mean_normalized
        
        # Policy Specific Parameters
        if self.policy == 'LB':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 1, 100, 1.0, 1
        elif self.policy == 'LD':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 2, 100, 1.0, 2
        elif self.policy == 'SM':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 15, 2, 70, 0.2, 2
        elif self.policy == 'SS':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 27, 2, 70, 0.2, 2
            
    # 시간 왜곡 함수 제거
    
    def timeMasking(self, feature):
        tau = feature.shape[2]  # time frames
        
        # apply m_T time masks to the mel spectrogram
        for i in range(self.m_T):
            t = int(np.random.uniform(0, self.T))  # [0, T)
            upper = tau if t > tau else t  # make limitation
            t0 = random.randint(0, tau - upper)  # [0, tau - t)
            feature[:, :, t0:t0 + t] = 0
            
        return feature

    def freqMasking(self, feature):
        size = feature.shape[1]
        
        for i in range(self.m_F):
            f = int(np.random.uniform(0, self.F))  # [0, F)
            f0 = random.randint(0, size - f)  # [0, v - f)
            feature[:, f0:f0 + f] = 0
            
        return feature
    
    def augment(self, feature):
        feature = self.freqMasking(feature)
        feature = self.timeMasking(feature)

        return feature
