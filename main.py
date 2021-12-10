import math
import time

import numpy as np
import random
import pprint
import matplotlib.pyplot as plt


class Car:
    time_unit = 1

    pos = None
    v = None  # (vx, vy)
    cur_section = None
    section_width = 2.5

    # bs = {choice_bs: bs}
    bs = None  # bs currently connecting, should clear on release
    # choice_bs = None  # this is a function pointer
    signal_powers = None
    is_just_handoff = None  # non-zero if just handoff (value for display different color for a while)

    # Deprecated, this is only for single algorithm #
    # this can be used to set to different policy.
    # choice_bs(car, bss) => bs

    calling_prob = 0  # call/second
    calling_interval_mean = 0  # seconds/call

    is_calling = False
    call_seconds = 0

    def __init__(self, pos, v, calling_prob, calling_interval_mean, choice_bs_set,
                 time_unit=time_unit):
        self.pos = np.array(pos)
        self.v = np.array(v)
        self.cur_section = self.get_section(self.pos)
        self.calling_prob = calling_prob
        self.calling_interval_mean = calling_interval_mean
        self.time_unit = time_unit
        self.choice_bs_set = choice_bs_set

        self.signal_powers = {}
        self.bs = {}
        self.is_just_handoff = {}
        # choice bs functions
        for cbs in choice_bs_set:
            self.bs[cbs] = None
            self.is_just_handoff[cbs] = 0

    def turn(self, is_left):
        self.v = np.cross((self.v[0], self.v[1], 0), (0, 0, -1 if is_left else 1))[0:2]
        # print(self.v)

    @staticmethod
    def turn_vec(v, is_left):
        return np.cross((v[0], v[1], 0), (0, 0, -1 if is_left else 1))[0:2]

    def update_call(self):
        # random by prob
        # if call, set call_seconds to seconds
        if self.is_calling:
            self.call_seconds -= self.time_unit
            if self.call_seconds <= 0:
                self.is_calling = False
                self.clear_bs()
                self.call_seconds = 0
            pass
        else:
            # normal distribution to call
            if random.choices([True, False], [self.calling_prob, 1 - self.calling_prob])[0]:
                # call
                self.is_calling = True
                self.call_seconds = np.random.normal(self.calling_interval_mean)

    def clear_bs(self):
        for cbs in self.choice_bs_set:
            self.bs[cbs] = None

    @staticmethod
    def get_section(pos, width=section_width):
        return np.array(pos) // width

    @staticmethod
    def next_pos(pos: np.ndarray, v: np.ndarray):
        # next pos after a time unit
        return pos + v

    def move(self, vel=None, width=section_width):
        # need to random turn if meet intersection
        if vel is None:
            vel = self.v
        cur_sec = self.cur_section
        new_pos = self.next_pos(self.pos, vel)
        nxt_sec = self.get_section(new_pos)
        self.cur_section = nxt_sec
        if (cur_sec != nxt_sec).any():
            # random turn
            nxt_pos = nxt_sec * width
            if vel[0] < 0:
                nxt_pos[0] += width
            if vel[1] < 0:
                nxt_pos[1] += width

            # remain_length = np.linalg.norm(new_pos - nxt_pos)
            remain_v = new_pos - nxt_pos
            self.pos = nxt_pos

            choice = random.choices([0, 1, 2, 3], [16 / 32, 2 / 32, 7 / 32, 7 / 32])[0]
            if choice == 0:
                # forward
                self.move(remain_v)
            elif choice == 1:
                # turn back
                self.v = self.v * -1
                self.move(remain_v)
            elif choice == 2:
                # turn left
                self.turn(is_left=True)
                self.move(self.turn_vec(remain_v, is_left=True))
            elif choice == 3:
                # turn right
                self.turn(is_left=False)
                self.move(self.turn_vec(remain_v, is_left=False))
        else:
            # not crossing intersection, keep going straight
            self.pos += self.v

    # handoff policy for cars
    # use lambda to set those custom parameters
    @staticmethod
    def policy_minimum(car, current_bs, min_sp):
        if current_bs is None:
            best_bs = max(car.signal_powers, key=car.signal_powers.get)
            return best_bs
        if car.signal_powers[current_bs] < min_sp:
            best_bs = max(car.signal_powers, key=car.signal_powers.get)
            return best_bs
        return current_bs

    @staticmethod
    def policy_best_effort(car, current_bs):
        return max(car.signal_powers, key=car.signal_powers.get)

    @staticmethod
    def policy_entropy(car, current_bs, ent_val):
        best_bs = max(car.signal_powers, key=car.signal_powers.get)
        if current_bs is None:
            return best_bs
        if (car.signal_powers[best_bs] - car.signal_powers[current_bs]) > ent_val:
            return best_bs
        return current_bs

    @staticmethod
    def policy_diy(car, current_bs):
        pass


class BaseStation:
    # car_count = {choice_bs: car_count}
    car_count = None

    def __init__(self, pos, freq, t_power):
        self.pos = np.array(pos)
        self.freq = freq
        self.t_power = t_power

        self.car_count = {}

    def __repr__(self):
        return f'bs pos: {self.pos}\n' \
               f'   freq: {self.freq}\n' \
               f'   t_power: {self.t_power}\n'


class Map:
    time_unit = 1  # seconds/frame
    car_in_lambda = 1 / 12  # car per second

    # calling parameters
    car_calling_prob = 2 / 60 / 60  # calls/second
    car_calling_interval_mean = 180  # seconds/call

    # car_in_entry_lambda = None  # prob for each entry
    # number of blocks
    exs = 10
    eys = 10
    width = 2.5

    car_v_val = (72 / 18 * 5) / 1000  # in km/s

    t_power = 120  # transmitting power

    car_choice_bs_set = [
        (lambda c, cur_bs: Car.policy_minimum(c, cur_bs, 100)),
        Car.policy_best_effort,
        (lambda c, cur_bs: Car.policy_entropy(c, cur_bs, 25))
    ]  # algorithm to choice bs by signals

    def __init__(self, time_unit=1):
        # convert time units
        self.car_in_lambda /= self.time_unit
        self.car_v_val *= self.time_unit

        # Length in km
        # freq in MHz
        self.cars = []
        self.bss = []

        self.signal_powers = {}  # {car: {bss: sp}}
        self.car_bss = {}  # which bss car is using

        self.border_x = self.exs * self.width
        self.border_y = self.eys * self.width

        self.entries = []  # entry points
        self.entry_v = {}  # initial velocity of each entry

        # due to numpy array is not hashable, store as tuple
        for ex in range(1, self.exs):
            self.entries.append((ex * self.width, 0))
            self.entry_v[self.entries[-1]] = (0, self.car_v_val)
        for ex in range(1, self.exs):
            self.entries.append((ex * self.width, self.eys * self.width))
            self.entry_v[self.entries[-1]] = (0, -self.car_v_val)

        for ey in range(1, self.eys):
            self.entries.append((0, ey * self.width))
            self.entry_v[self.entries[-1]] = (self.car_v_val, 0)
        for ey in range(1, self.eys):
            self.entries.append((self.exs * self.width, ey * self.width))
            self.entry_v[self.entries[-1]] = (-self.car_v_val, 0)
        # pprint.pprint(self.entries)

    def setup_bss(self, bss_counts=10, width=width):
        # random setup bss (list(BaseStation))
        bs_indexes = random.sample(range(self.exs * self.eys), bss_counts)
        freq = 100
        for index in bs_indexes:
            self.bss.append(BaseStation((index // self.eys * width + width / 2, index % self.eys * width + width / 2),
                                        freq, self.t_power))
            freq += 100
        return

    def next_frame(self):
        # pass 1 second
        self.poisson_generate_car()
        self.move_cars()
        self.remove_outside_cars()
        self.calculate_received_signal_powers()

        self.update_cars_call()
        return self.handoff()

    def poisson_generate_car(self):
        count = 0
        for en in self.entries:
            prob = self.poisson(self.car_in_lambda, 1)
            # print(random.choices([True, False], [prob, 1 - prob]))
            if random.choices([True, False], [prob, 1 - prob])[0]:
                new_car = Car((en[0], en[1]), list(self.entry_v[en]),
                              self.car_calling_prob, self.car_calling_interval_mean,
                              self.car_choice_bs_set)
                # new_car.choice_bs = self.car_choice_bs_function
                self.cars.append(new_car)
                count += 1
        return count

    def move_cars(self):
        for car in self.cars:
            car.move(car.v)

    def update_cars_call(self):
        for car in self.cars:
            car.update_call()

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
            car.signal_powers = {}
            for bs in self.bss:
                # self.signal_powers[car][bs] = self.received_signal_power(bs.t_power, car.pos, bs.pos, bs.freq)
                car.signal_powers[bs] = self.received_signal_power(bs.t_power, car.pos, bs.pos, bs.freq)

    def handoff(self):
        # handoff according to selected algorithm
        # choice bs function in Car class : choice_bs(car, bss) => bs
        handoff_count = {}
        for cbs in self.car_choice_bs_set:
            handoff_count[cbs] = 0
            for bs in self.bss:
                bs.car_count[cbs] = 0
        # for cbs in self.car_choice_bs_set:
        #     for car in self.cars:
        #         if car.bs[cbs] is not None:
        #             car.bs[cbs].car_count[cbs] += 1
        for car in self.cars:
            for cbs in self.car_choice_bs_set:
                if car.is_just_handoff[cbs] > 0:
                    # this value can be used to plot color to inform if this just handoff
                    car.is_just_handoff[cbs] -= self.time_unit

                if car.is_calling:
                    old_bs = car.bs[cbs]
                    new_bs = cbs(car, old_bs)
                    if new_bs is None:
                        exit(-1)

                    # car.bs should contain every choice_bs(lambda) as key
                    if old_bs is None:
                        # initial connect
                        car.bs[cbs] = new_bs
                    elif new_bs != old_bs:
                        # do handoff
                        # print('handoff:', old_bs.freq, new_bs.freq)
                        handoff_count[cbs] += 1
                        car.is_just_handoff[cbs] = 30
                        car.bs[cbs] = new_bs

                        car.bs[cbs].car_count[cbs] += 1
                    else:
                        # keep connecting
                        car.bs[cbs].car_count[cbs] += 1
                        pass

        return handoff_count

    @staticmethod
    def poisson(lmd, k):
        return math.exp(-1 * lmd) * math.pow(lmd, k) / math.factorial(k)

    @staticmethod
    def signal_path_loss(freq, dist):
        # freq(MHZ), dist(km)
        # print(freq, dist)
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
    # m.car_choice_bs_function = lambda c: Car.policy_minimum(c, 100)
    # pprint.pprint(m.bss)
    # exit(0)
    plt.ion()
    h_count = 0
    current_time = 0
    selected_choice_bs = m.car_choice_bs_set[1]
    # color setting
    car_color = 'steelblue'
    car_calling_color = 'red'
    # ticks
    xt = []
    yt = []
    x = 0
    base_xs = [b.pos[0] for b in m.bss]
    base_ys = [b.pos[1] for b in m.bss]
    while x <= 25:
        xt.append(x)
        yt.append(x)
        x += 2.5

    while True:
        start_time = time.perf_counter()
        current_time += 1
        h_count += m.next_frame()[selected_choice_bs]
        print('-' * 20)
        print('car count:', len(m.cars))
        print('handoff count:', h_count)

        xs = [c.pos[0] for c in m.cars]
        ys = [c.pos[1] for c in m.cars]

        colors = [(car_calling_color if c.is_calling else car_color) for c in m.cars]

        plt.subplot(121)
        plt.title(f'time: {current_time} s')
        plt.xticks(xt)
        plt.yticks(yt)
        plt.axis([0, 25, 0, 25])
        # plt.scatter(xs, ys, s=20)
        plt.scatter(xs, ys, s=20, c=colors)
        plt.scatter(base_xs, base_ys, s=30, c='orange')

        for c in m.cars:
            if c.is_calling:
                # print(c.pos, 'connect to', c.bs[selected_choice_bs].freq)
                plt.plot((c.pos[0], c.bs[selected_choice_bs].pos[0]),
                         (c.pos[1], c.bs[selected_choice_bs].pos[1]),
                         c=('yellow' if c.is_just_handoff[selected_choice_bs] == 0 else 'red'))

        bs_freqs = [b.freq for b in m.bss]
        for i, txt in enumerate(bs_freqs):
            plt.annotate(txt, (base_xs[i], base_ys[i]))
        plt.grid(True)
        # pprint.pprint(m.bss)
        plt.subplot(122)
        bs_index = [i for i in range(len(m.bss))]
        bs_service_counts = [b.car_count[selected_choice_bs] for b in m.bss]
        plt.xticks(bs_index)
        plt.bar(bs_index, bs_service_counts)
        print(bs_service_counts)

        plt.draw()
        # time.sleep(1)
        interval = 0.04
        print('pause:', (start_time + interval) - time.perf_counter())
        delay_time = (start_time + interval) - time.perf_counter()
        if delay_time > 0:
            plt.pause(delay_time)
        plt.pause(0.01)
        plt.clf()
