import gym
import numpy as np

class Car(object):
    def __init__(self, lane):
        self.pos = 0.
        self.v = 0.
        self.active = False
        self.lane = lane

    def clear(self):
        self.pos = 0.
        self.v = 0.
        self.active = False

    def move(self, time):
        self.pos += self.v * time

class Road(gym.Env):
    def __init__(self, N, M):
        #np.seed(0)
        self.car_num_max = N
        self.freq_num = M
        self.d_max = 120.
        self.d_min = 60.
        #self.road_len = self.d_max * (self.car_num_max - 1) + 50.
        self.road_len = self.d_min * (self.car_num_max - 1) + 50
        self.rho = 1.0 / 50.0
        self.car_dis_mean = 50.
        self.g_opposite = .2
        self.g_same = .02
        self.d_i = 40.                      # minimum interfering distance
        self.L = 8.
        sin_phi = self.L / np.sqrt(self.L ** 2 + 30. ** 2)
        self.P0 = 1e4
        self.N0 = 1/20 * self.P0 * self.g_opposite / (30. ** 2 + self.L ** 2) \
                  * (np.sin(0.5*np.pi*sin_phi)/(np.pi*sin_phi))**2
        self.d_same = np.sqrt(self.g_same*self.P0/self.N0)
        self.INR_upperbound = 10.
        self.INR_threshold = 1.
        self.v1_const = 30.
        self.v2_const = -25.
        self.time = 0.
        self.ts = .1

        self.cars_line1 = []
        self.cars_line2 = []
        for i in range(self.car_num_max):
            self.cars_line1.append(Car(0))
            self.cars_line2.append(Car(1))

    def _get_valid_car_dis(self):
        dis = np.random.exponential(self.car_dis_mean)
        while dis > self.d_max or dis < self.d_min:
            dis = np.random.exponential(self.car_dis_mean)
        return dis

    def _init_car_pos_v(self):
        '''
        Initialize the position of cars.
        :return:
        '''
        # initialize car positions in lane1
        cur_loc = 0
        car_idx = 0
        while cur_loc < self.road_len:
            self.cars_line1[car_idx].pos = cur_loc
            self.cars_line1[car_idx].v = self.v1_const
            self.cars_line1[car_idx].active = True
            cur_loc += self._get_valid_car_dis()
            car_idx += 1
        cur_loc = self.road_len
        car_idx = 0
        while cur_loc > 0:
            self.cars_line2[car_idx].pos = cur_loc
            self.cars_line2[car_idx].v = self.v2_const
            self.cars_line2[car_idx].active = True
            cur_loc -= self._get_valid_car_dis()
            car_idx += 1

    def _init_car_freq(self):
        '''
        Initialize the freq band of cars.
        :return:
        '''
        raise NotImplementedError

    def _observe(self):
        '''
        Form observation of current state.
        :return:
        '''

    def _get_reward(self):
        '''
        Get the reward of each car.
        :return:
        '''
    def reset(self):
        '''
        Reset Road environment and return the observation of the initiate state.
        :return:
        '''
        raise NotImplementedError

    def step(self, action):
        '''
        Step function.
        :param action:
        :return:
        '''
        raise NotImplementedError


    








