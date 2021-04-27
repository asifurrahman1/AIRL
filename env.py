import gym
gym.logger.set_level(40)
from env_process import process_state_image

def make_env(env_id):
    return NormalizedEnv(gym.make(env_id))

class NormalizedEnv(gym.Wrapper):

    def __init__(self, env):
        self.observation_shape=(96,96)
        gym.Wrapper.__init__(self, env)
    
    def reset(self):
        s = self.env.reset()
        s = process_state_image(s)
        # print(s.shape)
        return s
        
    def step(self, action):
        s,r,d,ot = self.env.step(action)
        s = process_state_image(s)
        # print(s.shape)
        return s,r,d,ot