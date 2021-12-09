import math
import time

import numpy as np
import random
import pprint
import matplotlib.pyplot as plt


class Car:
    pos = None
    v = None  # (vx, vy)
    section_width = 2.5

    bs = None  # bs currently connecting, should clear on release
    choice_bs = None  # this is a function pointer
    signal_powers = {}
    # choice_bs(car, bss) => bs
    # this can be used to set to different policy.

    is_calling = False
    call_seconds = 0

    def __init__(self, pos, v):
        self.pos = np.array(pos)
        self.v = np.array(v)

    def turn(self, is_left):
        self.v = np.cross((self.v[0], self.v[1], 0), (0, 0, -1 if is_left else 1))

    def update_call(self, prob, seconds=30):
        # random by prob
        # if call, set call_seconds to seconds
        if self.is_calling:
            self.call_seconds -= 1
            if self.call_seconds <= 0:
                self.is_calling = False
            pass
        else:
            # normal distribution to call
            pass

    @staticmethod
    def get_section(pos, width=section_width):
        return np.array(pos) // width

    @staticmethod
    def next_pos(pos: np.ndarray, v: np.ndarray):
        # next pos after a time unit
        return pos + v

    def move(self, length=None, width=section_width):
        # need to random turn if meet intersection
        if length is None:
            length = self.v
        cur_sec = self.get_section(self.pos)
        new_pos = self.next_pos(self.pos, length)
        nxt_sec = self.get_section(new_pos)
        if (cur_sec != nxt_sec).all():
            # random turn
            nxt_pos = nxt_sec * width
            remain_length = np.linalg.norm(new_pos - nxt_pos)
            self.pos = nxt_pos

            choice = random.choices([0, 1, 2, 3], [16 / 32, 2 / 32, 7 / 32, 7 / 32])
            if choice == 0:
                # forward
                self.move(remain_length)
            elif choice == 1:
                # turn back
                self.v = self.v * -1
                self.move(remain_length)
            elif choice == 2:
                # turn left
                self.turn(is_left=True)
                self.move(remain_length)
            elif choice == 3:
                # turn right
                self.turn(is_left=False)
                self.move(remain_length)
        else:
            # not crossing intersection, keep going straight
            self.pos += self.v

    # handoff policy for cars
    # use lambda to set those custom parameters
    @staticmethod
    def policy_minimum(car, min_sp):
        if not car.bs:
            best_bs = max(car.signal_powers, key=car.get)
            return best_bs
        if car.signal_powers[car.bs] < min_sp:
            best_bs = max(car.signal_powers, key=car.get)
            return best_bs
        return car.bs

    @staticmethod
    def policy_best_effort(car):
        return max(car.signal_powers, key=car.get)

    @staticmethod
    def policy_entropy(car, ent_val):
        best_bs = max(car.signal_powers, key=car.get)
        if not car.bs:
            return best_bs
        if (car.signal_powers[best_bs] - car.signal_powers[car.bs]) > ent_val:
            return best_bs
        return car.bs

    @staticmethod
    def policy_diy(car):
        pass


class BaseStation:
    def __init__(self, pos, freq, t_power):
        self.pos = np.array(pos)
        self.freq = freq
        self.t_power = t_power


class Map:
    car_in_lambda = 0.01  # car per second
    # number of blocks
    exs = 10
    eys = 10
    width = 2.5

    t_power = 120

    car_choice_bs_function = None

    def __init__(self):
        # Length in km
        # freq in MHz
        self.cars = []
        self.bss = []

        self.signal_powers = {}  # {car: {bss: sp}}
        self.car_bss = {}  # which bss car is using

        self.setup_bss()

        self.border_x = self.exs * self.width
        self.border_y = self.eys * self.width

        self.entries = []  # entry points
        self.entry_v = {}  # initial velocity of each entry

        # due to numpy array is not hashable, store as tuple
        for ex in range(1, self.exs):
            self.entries.append((ex * self.width, 0))
            self.entry_v[self.entries[-1]] = (0, 1)
        for ex in range(1, self.exs):
            self.entries.append((ex * self.width, self.eys * self.width))
            self.entry_v[self.entries[-1]] = (0, -1)

        for ey in range(1, self.eys):
            self.entries.append((0, ey * self.width))
            self.entry_v[self.entries[-1]] = (1, 0)
        for ey in range(1, self.eys):
            self.entries.append((self.exs * self.width, ey * self.width))
            self.entry_v[self.entries[-1]] = (-1, 0)
        # pprint.pprint(self.entries)

    def setup_bss(self, bss_counts=10, width=width):
        # random setup bss (list(BaseStation))
        bs_indexes = random.sample(range(self.exs * self.eys), bss_counts)
        for index in bs_indexes:
            self.bss.append(BaseStation((index // self.eys * width, index % self.eys), (index + 1) * 100, self.t_power))
        return

    def next_frame(self):
        # pass 1 second
        self.poisson_generate_car()
        self.move_cars()
        self.remove_outside_cars()
        self.calculate_received_signal_powers()
        return self.handoff()

    def poisson_generate_car(self):
        count = 0
        for en in self.entries:
            prob = self.poisson(self.car_in_lambda, 1)
            if random.choices([True, False], [prob, 1 - prob]):
                new_car = Car((en[0], en[1]), list(self.entry_v[en]))
                new_car.choice_bs = self.car_choice_bs_function
                self.cars.append(new_car)
                count += 1
        return count

    def move_cars(self):
        for car in self.cars:
            car.move(car.v)

    def remove_outside_cars(self):
        remain_cars = []
        outs = 0
        for car in self.cars:
            if not (car.pos[0] < 0 or car.pos[0] > self.border_x or car.pos[1] < 0 or car.pos[1] > self.border_y):
                remain_cars.append(car)
            else:
                outs += 1
        self.cars = remain_cars
        return outs

    def calculate_received_signal_powers(self):
        for car in self.cars:
            # self.signal_powers[car] = {}
            car.signal_powers = {}
            for bs in self.bss:
                # self.signal_powers[car][bs] = self.received_signal_power(bs.t_power, car.pos, bs.pos, bs.freq)
                car.signal_powers[bs] = self.received_signal_power(bs.t_power, car.pos, bs.pos, bs.freq)

    def handoff(self):
        # handoff according to selected algorithm
        # choice bs function in Car class : choice_bs(car, bss) => bs
        handoff_count = 0
        for car in self.cars:
            if car.is_calling:
                new_bs = car.choice_bs(car, self.bss)
                if car.bs is None:
                    car.bs = new_bs
                elif new_bs != car.bs:
                    handoff_count += 1
                    car.bs = new_bs
        return handoff_count

    @staticmethod
    def poisson(lmd, k):
        return math.exp(-1 * lmd) * math.pow(lmd, k) / math.factorial(k)

    @staticmethod
    def signal_path_loss(freq, dist):
        # freq(MHZ), dist(km)
        print(freq, dist)
        if dist != 0:
            return 32.45 + 20 * math.log10(freq) + 20 * math.log10(dist)
        else:
            return 32.45 + 20 * math.log10(freq)

    def received_signal_power(self, pt, pos1, pos2, freq):
        # pt = Transmitting power (dB)
        # freq = Transmitting frequency (MHz)
        # pos1, pos2 (km)
        return pt - self.signal_path_loss(freq, abs(np.linalg.norm(pos1 - pos2)))


if __name__ == '__main__':
    print('hello')
    m = Map()
    m.setup_bss(10, 2.5)
    m.car_choice_bs_function = lambda c: Car.policy_minimum(c, 100)
    while True:
        handoff_count = m.next_frame()
        print('-' * 20)
        print('car count:', len(m.cars))
        print('handoff count:', handoff_count)
        time.sleep(1)
