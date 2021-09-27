import gym
import numpy as np
import queue
import matplotlib.pyplot as plt


class Car(object):
    def __init__(self, lane, car_id=None):
        self.pos = 0.
        self.v = 0.
        self.active = False
        self.f = 0                                                   #initial freq band of this car is 0.
        self.I = 0.
        self.R = 0.
        self.lane = lane
        self.car_id = car_id

    def clear(self):    # This function is not safe temporarily..
        self.pos = 0.
        self.v = 0.
        self.I = 0.
        self.active = False
        self.f = 0
        self.R = 0.

    def move(self, time):
        if self.active:
            self.pos += self.v * time


class Road(gym.Env):
    def __init__(self, N, M):
        #np.seed(0)
        self.car_num_max = N
        self.freq_num = M
        self.observation_dim = 9
        self.d_max = 100.
        self.d_min = 60.
        self.max_pass_cars_num = 30
        self.road_len = self.d_min * (self.car_num_max - 1) + 50.
        #self.rho = 1.0 / 50.0
        self.car_dis_mean = 50.
        self.g_opposite = .1
        self.g_same = .02
        self.d_i = 40.                                                                # minimum interfering distance
        self.L = 8.
        sin_phi = self.L / np.sqrt(self.L ** 2 + 30. ** 2)
        self.P0 = 1e4
        self.N0 = 1/10 * self.P0 * self.g_opposite / (30. ** 2 + self.L ** 2) \
                  * (np.sin(0.5*np.pi*sin_phi)/(np.pi*sin_phi))**2
        self.d_same = np.sqrt(self.g_same*self.P0/self.N0)
        self.INR_upperbound = 10.
        self.INR_threshold = 1.
        self.v1_const = 30.
        self.v2_const = -25.
        self.r_var = 0.1
        self.time = 0.
        self.ts = .1
        self.stop_time = 2 * self.road_len / abs(self.v2_const)
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
        self.next_car_interval_line1 = None
        self.next_car_interval_line2 = None
        self.is_completed = None
        self.alive_mask = None
        for i in range(self.car_num_max):
            self.cars_line1.append(Car(1, i))
            self.cars_line2.append(Car(2, i))

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

        # set next car interval
        self.next_car_interval_line1 = self.cars_interval_line1.get()
        self.next_car_interval_line2 = self.cars_interval_line2.get()

    def _get_last_car_idx(self, lane):
        idx = 0
        if lane == 1:
            last_dis = 0
            for i, car in enumerate(self.cars_line1):
                if car.active and car.pos >= last_dis:
                    last_dis = car.pos
                    idx = i
        if lane == 2:
            last_dis = self.road_len
            for i, car in enumerate(self.cars_line2):
                if car.active and car.pos <= last_dis:
                    last_dis = car.pos
                    idx = i
        return idx

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
        self.car_end_idx_line1 = self._get_last_car_idx(1)
        self.car_end_idx_line2 = self._get_last_car_idx(2)
        assert self.cars_line1[self.car_end_idx_line1].active
        assert self.cars_line2[self.car_end_idx_line2].active
        if self.cars_line1[self.car_end_idx_line1].pos > self.road_len:
            self.cars_line1_wait_q.put(self.car_end_idx_line1)
            self.cars_line1[self.car_end_idx_line1].clear()
            self.is_completed[self.car_end_idx_line1] = 1
        if self.cars_line2[self.car_end_idx_line2].pos < 0:
            self.cars_line2_wait_q.put(self.car_end_idx_line2)
            self.cars_line2[self.car_end_idx_line2].clear()
            self.is_completed[self.car_num_max + self.car_end_idx_line2] = 1
        # add cars into road
        assert self.cars_line1[self.car_begin_idx_line1].active
        assert self.cars_line2[self.car_begin_idx_line2].active
        if self.cars_line1[self.car_begin_idx_line1].pos > self.next_car_interval_line1:
            self.car_begin_idx_line1 = self.cars_line1_wait_q.get()
            self.cars_line1[self.car_begin_idx_line1].active = True
            self.cars_line1[self.car_begin_idx_line1].pos = 0.
            self.cars_line1[self.car_begin_idx_line1].v = self.v1_const
            self.cars_line1[self.car_begin_idx_line1].f = 0
            self.next_car_interval_line1 = self.cars_interval_line1.get()
        if self.cars_line2[self.car_begin_idx_line2].pos < self.road_len - self.next_car_interval_line2:
            self.car_begin_idx_line2 = self.cars_line2_wait_q.get()
            self.cars_line2[self.car_begin_idx_line2].active = True
            self.cars_line2[self.car_begin_idx_line2].pos = self.road_len
            self.cars_line2[self.car_begin_idx_line2].v = self.v2_const
            self.cars_line2[self.car_begin_idx_line2].f = 0
            self.next_car_interval_line2 = self.cars_interval_line2.get()
        #update alive car mask
        for idx, car in enumerate(self.cars_line1):
            if car.active:
                self.alive_mask[idx] = 1
        for idx, car in enumerate(self.cars_line2):
            if car.active:
                self.alive_mask[self.car_num_max + idx] = 1

    def _calculate_interference(self):
        # calculate the interference the cars in the lane1 receive
        for i, this_car in enumerate(self.cars_line1):
            if this_car.active:
                this_car.I = 0.
                for j, car in enumerate(self.cars_line1):
                    if car.active:
                        if (i != j and this_car.f == car.f and car.pos - this_car.pos > 0):
                            this_car.I += self.P0 * self.g_same / (car.pos - this_car.pos)**2
                for j, car in enumerate(self.cars_line2):
                    if car.active:
                        if (car.f == this_car.f and car.pos - this_car.pos > self.d_i):
                            sin_phi = self.L / np.sqrt(self.L ** 2 + (car.pos - this_car.pos) ** 2)
                            this_car.I += self.P0 * self.g_opposite / ((car.pos - this_car.pos) ** 2 + self.L**2)\
                                                 * (np.sin(0.5*np.pi*sin_phi)/(np.pi*sin_phi))**2
                this_car.I = this_car.I / self.N0
                this_car.R = float(this_car.I <= self.INR_threshold)
        # calculate the interference the cars in the lane2 receive
        for i, this_car in enumerate(self.cars_line2):
            if this_car.active:
                this_car.I = 0.
                for j, car in enumerate(self.cars_line2):
                    if car.active:
                        if (i != j and this_car.f == car.f and car.pos - this_car.pos < 0):
                            this_car.I += self.P0 * self.g_same / (car.pos - this_car.pos) ** 2
                for j, car in enumerate(self.cars_line1):
                    if car.active:
                        if (car.f == this_car.f and this_car.pos - car.pos > self.d_i):
                            sin_phi = self.L / np.sqrt(self.L ** 2 + (car.pos - this_car.pos) ** 2)
                            this_car.I += self.P0 * self.g_opposite / ((car.pos - this_car.pos) ** 2 + self.L**2)\
                                                 * (np.sin(0.5*np.pi*sin_phi)/(np.pi*sin_phi))**2
                this_car.I = this_car.I / self.N0
                this_car.R = float(this_car.I <= self.INR_threshold)

    def _real_pos(self, idx, line):
        if line == 1:
            pos_same, pos_oppo = self.road_len + 1., self.road_len + 1.
            this_car = self.cars_line1[idx]
            if this_car.active:
                for i, car in enumerate(self.cars_line1):
                    if (i != idx and car.pos > this_car.pos and pos_same > car.pos):
                        pos_same = car.pos
                for i, car in enumerate(self.cars_line2):
                    if (car.pos - this_car.pos > self.d_i and pos_oppo > car.pos):
                        pos_oppo = car.pos
        else:
            pos_same, pos_oppo = -1., -1.
            this_car = self.cars_line2[idx]
            if this_car.active:
                for i, car in enumerate(self.cars_line2):
                    if (i != idx and car.pos < this_car.pos and pos_same < car.pos):
                        pos_same = car.pos
                for i, car in enumerate(self.cars_line1):
                    if (this_car.pos - car.pos > self.d_i and pos_oppo < car.pos):
                        pos_oppo = car.pos
        return pos_same, pos_oppo

    def _car_state(self, idx, line):
        if line == 1:
            this_car = self.cars_line1[idx]
            if this_car.active:
                pos_same, pos_oppo = self._real_pos(idx, line)
                if pos_same <= self.road_len:
                    pos_same = pos_same + this_car.I * np.sqrt(self.r_var) * np.random.randn(1)[0]
                if pos_oppo <= self.road_len:
                    pos_oppo = pos_oppo + this_car.I * np.sqrt(self.r_var) * np.random.randn(1)[0]
                this_car_norm_i = np.min((this_car.I, self.INR_upperbound)) / self.INR_upperbound
                s = np.array([this_car.f / self.freq_num, this_car_norm_i, this_car.R,  this_car.pos / self.road_len,
                              pos_same / self.road_len, pos_oppo / self.road_len, this_car.car_id / self.car_num_max,
                              0, 1])
            else:
                s = np.array([0, 0, 0, 0, 0, 0, this_car.car_id / self.car_num_max, 0, 0])
        else:
            this_car = self.cars_line2[idx]
            if this_car.active:
                pos_same, pos_oppo = self._real_pos(idx, line)
                if pos_same >= 0:
                    pos_same = pos_same + this_car.I * np.sqrt(self.r_var) * np.random.randn(1)[0]
                if pos_oppo >= 0:
                    pos_oppo = pos_oppo + this_car.I * np.sqrt(self.r_var) * np.random.randn(1)[0]
                this_car_norm_i = np.min((this_car.I, self.INR_upperbound)) / self.INR_upperbound
                s = np.array([this_car.f / self.freq_num, this_car_norm_i, this_car.R, this_car.pos / self.road_len,
                              pos_same / self.road_len, pos_oppo / self.road_len, this_car.car_id / self.car_num_max,
                              1, 1])
            else:
                s = np.array([0, 0, 0, 1, 1, 1, this_car.car_id / self.car_num_max, 1, 0])
        return s

    def _observe(self):
        '''
        Form observation of current state.
        :return:
        '''
        obs = np.zeros((self.car_num_max * 2, 9), dtype=np.float)
        for i in range(self.car_num_max):
            obs[i, :] = self._car_state(i, 1)
        for i in range(self.car_num_max):
            obs[i + self.car_num_max, :] = self._car_state(i, 2)
        return obs[np.newaxis, :]

    def _get_reward(self):
        '''
        Get the reward of each car.
        :return:
        '''
        #num_active_car = 0
        #sum_rew = 0
        rew = np.zeros(2 * self.car_num_max)
        # for car in self.cars_line1:
        #     if car.active:
        #         num_active_car += 1
        #         sum_rew += car.R
        # for car in self.cars_line2:
        #     if car.active:
        #         num_active_car += 1
        #         sum_rew += car.R
        for i, car in enumerate(self.cars_line1):
            rew[i] = car.R
        for i, car in enumerate(self.cars_line2):
            rew[i + self.car_num_max] = car.R
        return rew

    def _clear_queue(self):
        while not self.cars_line1_wait_q.empty():
            self.cars_line1_wait_q.get()
        while not self.cars_line2_wait_q.empty():
            self.cars_line2_wait_q.get()
        while not self.cars_interval_line1.empty():
            self.cars_interval_line1.get()
        while not self.cars_interval_line2.empty():
            self.cars_interval_line2.get()

    def _clear_car_state(self):
        for car in self.cars_line1:
            car.clear()
        for car in self.cars_line2:
            car.clear()

    def reset(self):
        '''
        Reset Road environment and return the observation of the initiate state.
        :return:
        '''
        self._clear_queue()
        self._clear_car_state()
        self.car_begin_idx_line1 = 0
        self.car_begin_idx_line2 = 0
        self.car_end_idx_line1 = None
        self.car_end_idx_line2 = None
        self.next_car_interval_line1 = None
        self.next_car_interval_line2 = None
        self.is_completed = np.zeros(2 * self.car_num_max)
        self.alive_mask = np.zeros(2 * self.car_num_max)
        self._get_car_dist_list()
        self._init_car_pos_v()
        self.time = 0.
        self._calculate_interference()
        return self._observe()

    def step(self, action):
        '''
        Step function.
        :param action: np.array(2*N)
        :return:
        '''
        # step 1: set freq of cars
        for i, car in enumerate(self.cars_line1):
            car.f = int(action[i])
        for i, car in enumerate(self.cars_line2):
            car.f = int(action[i + self.car_num_max])
        # step 2:move cars
        self.is_completed = np.zeros(2 * self.car_num_max)
        self._move_cars()
        self.time += self.ts
        # step 3:calculate interference
        self._calculate_interference()
        # obs
        obs = self._observe()
        rew = self._get_reward()
        done = True if self.time > self.stop_time else False
        num_cars_1, num_cars_2 = 0, 0
        for car in self.cars_line1:
            num_cars_1 += (1 if car.active else 0)
        for car in self.cars_line2:
            num_cars_2 += (1 if car.active else 0)
        info = {'num_cars_line1': num_cars_1, 'num_cars_line2': num_cars_2, 'alive_mask': self.alive_mask,
                'is_completed': self.is_completed}
        return obs, rew, done, info

    def render(self, mode='human'):
        marker = ['^', 'o']
        color = ['red', 'green', 'blue', 'purple']
        plt.ion()
        plt.plot(np.linspace(0, self.road_len, 1000), 2.5 * np.ones(1000), 'black')
        plt.plot(np.linspace(0, self.road_len, 1000), 1.5 * np.ones(1000), 'black')
        plt.plot(np.linspace(0, self.road_len, 1000), 0.5 * np.ones(1000), 'black')

        for i in range(self.car_num_max):
            this_car = self.cars_line1[i]
            if this_car.active:
                plt.plot(this_car.pos, 2, color=color[int(this_car.f)], marker=marker[int(this_car.R)], markersize=10)

        for i in range(self.car_num_max):
            this_car = self.cars_line2[i]
            if this_car.active:
                plt.plot(this_car.pos, 1, color=color[int(this_car.f)], marker=marker[int(this_car.R)], markersize=10)

        plt.xlim(0-50, self.road_len+50)
        plt.ylim(0.4-3, 2.6+3)
        plt.pause(1)
        plt.close()









    








