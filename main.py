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
    def next_pos(pos, v):
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
            over_length = np.linalg.norm(new_pos - nxt_pos)
            self.pos = nxt_pos
            choice = random.choices([0, 1, 2, 3], [16 / 32, 2 / 32, 7 / 32, 7 / 32])
            if choice == 0:
                # forward
                self.move(over_length)
            elif choice == 1:
                # turn back
                self.v = self.v * -1
            elif choice == 2:
                # turn left
                self.turn(is_left=True)
                self.move(over_length)
            elif choice == 3:
                # turn right
                self.turn(is_left=False)
                self.move(over_length)

        else:
            self.pos += self.v


class Map:

    def __init__(self):
        self.cars = []
        self.bss = []
