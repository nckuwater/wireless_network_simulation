import math
import numpy as np
import random


class Car:
    pos = None
    v = None  # (vx, vy)
    section_width = 2.5

    def __init__(self, pos, v):
        self.pos = np.array(pos)
        self.v = np.array(v)

    def turn(self, is_left):
        self.v = np.cross((self.v[0], self.v[1], 0), (0, 0, -1 if is_left else 1))

    @staticmethod
    def get_section(pos, width=section_width):
        return np.array(pos) // width

    @staticmethod
    def next_pos(pos: np.ndarray, v: np.ndarray):
        return pos + v

    def move(self, length=None, width=section_width):
        # need to random turn if meet intersection
        if length is None:
            length = self.v
        cur_sec = self.get_section(self.pos)
        new_pos = self.next_pos(self.pos, length)
        nxt_sec = self.get_section(new_pos)
        if cur_sec != nxt_sec:
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
            self.pos += self.v


class Map:
    car_in_lambda = 0.01  # car per second
    # number of entries
    exs = 10
    eys = 10
    width = 2.5

    def __init__(self):
        self.cars = []
        self.bss = []
        self.bss_freq = {}
        self.signal_powers = {}  # {car: {bss: sp}}
        self.car_bss = {}  # which bss car is using

        self.setup_bss()

        self.border_x = self.exs * self.width
        self.border_y = self.eys * self.width

        self.entries = []
        self.entry_v = {}

        for ex in range(1, self.exs):
            self.entries.append(np.array(ex, 0) * self.width)
            self.entry_v[self.entries[-1]] = (0, 1)
        for ex in range(1, self.exs):
            self.entries.append(np.array(ex, self.eys) * self.width)
            self.entry_v[self.entries[-1]] = (0, -1)

        for ey in range(1, self.eys):
            self.entries.append(np.array(0, ey) * self.width)
            self.entry_v[self.entries[-1]] = (1, 0)
        for ey in range(1, self.eys):
            self.entries.append(np.array(self.exs, ey) * self.width)
            self.entry_v[self.entries[-1]] = (-1, 0)

    def setup_bss(self):
        return

    def next_frame(self):
        # pass 1 second
        self.poisson_generate_car()
        self.move_cars()
        self.remove_outside_cars()
        self.calculate_received_signal_powers()
        self.check_handoff()

    def poisson_generate_car(self):
        count = 0
        for en in self.entries:
            prob = self.poisson(self.car_in_lambda, 1)
            if random.choices([True, False], [prob, 1 - prob]):
                self.cars.append(Car((en[0], en[1]), self.entry_v[en]))
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
            self.signal_powers[car] = {}
            for bs in self.bss:
                self.signal_powers[car][bs] = self.received_signal_power(car.pos, bs.pos, self.bss_freq[bs])

    def check_handoff(self):
        # handoff according to selected algorithm
        pass

    @staticmethod
    def poisson(lmd, k):
        return math.exp(-1 * lmd) * math.pow(lmd, k) / math.factorial(k)

    @staticmethod
    def signal_power(freq, dist):
        return 32.45 + 20 * math.log10(freq) + 20 * math.log10(dist)

    def received_signal_power(self, pos1, pos2, freq):
        return self.signal_power(freq, np.linalg.norm(pos1, pos2))


if __name__ == '__main__':
    print('hello')
