import gym
import numpy as np
import queue

class Car(object):
    def __init__(self, lane, car_id=None):
        self.pos = 0.
        self.v = 0.
        self.active = False
        self.f = 0                                                   #initial freq band of this car is 0.
        self.lane = lane
        self.car_id = car_id

    def clear(self):    # This function is not safe temporarily..
        self.pos = 0.
        self.v = 0.
        self.active = False
        self.f = 0

    def move(self, time):
        if self.active:
            self.pos += self.v * time

class Road(gym.Env):
    def __init__(self, N, M):
        #np.seed(0)
        self.car_num_max = N
        self.freq_num = M
        self.d_max = 120.
        self.d_min = 60.
        self.max_pass_cars_num = 100
        self.road_len = self.d_min * (self.car_num_max - 1) + 50.
        self.rho = 1.0 / 50.0
        self.car_dis_mean = 50.
        self.g_opposite = .2
        self.g_same = .02
        self.d_i = 40.                                              # minimum interfering distance
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
        self.cars_line1_wait_q = queue.Queue(maxsize=self.car_num_max)              #todo: Clear the following 4 queue while env is done.
        self.cars_line2_wait_q = queue.Queue(maxsize=self.car_num_max)
        self.cars_interval_line1 = queue.Queue(maxsize=self.max_pass_cars_num)
        self.cars_interval_line2 = queue.Queue(maxsize=self.max_pass_cars_num)
        self.car_begin_idx_line1 = 0
        self.car_begin_idx_line2 = 0
        self.car_end_idx_line1 = None
        self.car_end_idx_line2 = None
        self._get_car_dist_list()
        for i in range(self.car_num_max):
            self.cars_line1.append(Car(0, i))
            self.cars_line2.append(Car(1, i))

    def _get_car_dist_list(self):
        for i in range(self.max_pass_cars_num):
            self.cars_interval_line1.put(self._get_valid_car_dis())
            self.cars_interval_line2.put(self._get_valid_car_dis())

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
            cur_loc += self.cars_interval_line1.get()
            self.car_end_idx_line1 = car_idx
            car_idx += 1
        # put inactive car in line1 in waiting queue
        while car_idx < self.car_num_max:
            self.cars_line1_wait_q.put(car_idx)
            car_idx += 1

        # initialize car positions in lane2
        cur_loc = self.road_len
        car_idx = 0
        while cur_loc > 0:
            self.cars_line2[car_idx].pos = cur_loc
            self.cars_line2[car_idx].v = self.v2_const
            self.cars_line2[car_idx].active = True
            cur_loc -= self.cars_interval_line2.get()
            self.car_end_idx_line2 = car_idx
            car_idx += 1
        # put inactive car in line2 in waiting queue
        while car_idx < self.car_num_max:
            self.cars_line2_wait_q.put(car_idx)
            car_idx += 1

    def _move_cars(self):
        '''
        move cars a step forward in 2 lines.
        :return:
        '''
        # move active cars a step forward
        for car in self.cars_line1:
            car.move(self.ts)
        for car in self.cars_line2:
            car.move(self.ts)
        # remove cars out of road range
        assert self.cars_line1[self.car_end_idx_line1].active
        assert self.cars_line2[self.car_end_idx_line2].active
        # 这里想把正在路上的车的信息也存成队列，但是python队列没有 查看队列头部元素的能力。。。是不是应该换一种数据结构？
        #if self.cars_line1[self.car_end_idx_line1].pos > self.road_len:


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


    








