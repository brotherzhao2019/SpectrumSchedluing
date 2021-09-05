import gym

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

