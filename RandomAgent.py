from RoadEnv import Road
import numpy as np

class RandomAgent(object):
    def __init__(self, N, M):
        self.num_freq = M
        self.num_car = N
    def get_action(self, obs:np.ndarray):
        return np.random.randint(0, self.num_freq, 2 * self.num_car)

def eval_random_agent(road_env:Road, random_agent:RandomAgent):
    #road_env = Road(5, 3)
    #random_agent = RandomAgent(5, 3)
    obs = []
    rews = []
    acts = []
    dones = []
    ob = None
    obtp1 = None
    rew = None
    act = None
    done = False
    ob = road_env.reset()
    while not done:
        act = random_agent.get_action(ob)
        obtp1, rew, done, info = road_env.step(act)
        road_env.render()
        obs.append(ob)
        acts.append(act)
        rews.append(rew)
        dones.append(done)
        ob = obtp1
    print("average reward: {}".format(np.mean(rews)))

def main():
    road_env = Road(5, 3)
    random_agent = RandomAgent(5, 3)
    eval_random_agent(road_env, random_agent)

if __name__ == '__main__':
    main()



