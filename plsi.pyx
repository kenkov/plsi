#! /usr/bin/env python
# coding:utf-8

from collections import defaultdict
import random


def normalize(dic):
    cdef float total = 0
    cdef double val
    for key, val in dic.items():
        total += val
    for key, val in dic.items():
        dic[key] = val / total


def train(
    file_name: str,
    num_z: int,
    loop_count: int
):

    cdef long num_d = 0
    cdef long z, d
    n_d_w = defaultdict(int)
    word_set = set()

    with open(file_name) as f:
        for d, line in enumerate((line.strip() for line in f)):
            num_d += 1
            words = line.split()
            for w in words:
                n_d_w[(d, w)] += 1
                word_set.add(w)

    p_z_dw = defaultdict(float)
    p_w_z = defaultdict(float)
    p_d_z = defaultdict(float)
    p_z = defaultdict(float)

    # init
    for z in range(num_z):
        p_z[z] = random.random()
        for d in range(num_d):
            for w in word_set:
                    p_w_z[(w, z)] = random.random()
                    p_d_z[(d, z)] = random.random()
    # normalize
    print(p_z)
    normalize(p_z)
    normalize(p_w_z)
    normalize(p_d_z)

    for _ in range(loop_count):
        # E step
        for d in range(num_d):
            for w in word_set:
                denom_z = sum(
                    p_z[z] * p_d_z[(d, z)] * p_w_z[(w, z)]
                    for z in range(num_z)
                )
                for z in range(num_z):
                    p_z_dw[(z, d, w)] = \
                        p_z[z] * p_d_z[(d, z)] * p_w_z[(w, z)] / denom_z
        # M step
        for z in range(num_z):
            denom_d_w = sum(
                n_d_w[(d, w)] * p_z_dw[(z, d, w)]
                for d in range(num_d) for w in word_set
            )
            denom_z = sum(
                n_d_w[(d, w)]
                for d in range(num_d) for w in word_set
            )
            for d in range(num_d):
                for w in word_set:
                    p_w_z[(w, z)] = sum(
                        n_d_w[(d, w)] * p_z_dw[(z, d, w)]
                        for d in range(num_d)
                    ) / denom_d_w
                    p_d_z[(d, z)] = sum(
                        n_d_w[(d, w)] * p_z_dw[(z, d, w)]
                        for w in word_set
                    ) / denom_d_w
                    p_z[z] = sum(
                        n_d_w[(d, w)] * p_z_dw[(z, d, w)]
                        for d in range(num_d) for w in word_set
                    ) / denom_z

    for (z, d, w), val in p_z_dw.items():
        print("{} {} {}\t{:.6f}".format(
            z, d, w, val
        ))
    print()
    for (w, z), val in p_w_z.items():
        print("{} {}\t{:.6f}".format(w, z, val))
    print()
    for (d, z), val in p_d_z.items():
        print("{} {}\t{:.6f}".format(d, z, val))
    print()
    for z, val in p_z.items():
        print("{}\t{:.6f}".format(z, val))
