#! /usr/bin/env python
# coding:utf-8

from collections import defaultdict
import numpy as np
from logging import getLogger


class Corpus():
    def __init__(self, filename, dictionary):
        self.filename = filename
        self.dictionary = dictionary

    def __iter__(self):
        for line in (line.strip() for line in open(self.filename)):
            yield [self.dictionary[word] for word in line.split()]


def mkdictionary(xs: [[str]]) -> {str: int}:
    _id = 0
    dic = dict()
    for line in xs:
        for word in line:
            if word not in dic:
                dic[word] = _id
                _id += 1
    return dic


def file2dic(filename):
    return mkdictionary(
        line.strip().split() for line in open(args.filename)
    )


def data(corpus):
    for d, line in enumerate(corpus):
        for w in line:
            yield (d, w)


class PLSI:
    def __init__(
        self,
        corpus,
        dictionary
    ):
        self.corpus = corpus
        self.dictionary = dictionary

    def load(
        self,
        modelfile,
        logger=None
    ):

        if not logger:
            logger = getLogger(__file__)

        fd = open(modelfile)
        for line in (_.strip() for _ in fd):
            if line == "":
                break
            else:
                num_d, num_w, num_z = [int(num) for num in line.split()]

        logger.info("load model: num_d {}, num_w {}, num_z {}".format(
            num_d, num_w, num_z
        ))
        # define instance variables
        self.num_d = num_d
        self.num_w = num_w
        self.num_z = num_z

        p_z = np.zeros(num_z)
        p_w_z = np.zeros((num_w, num_z))
        p_d_z = np.zeros((num_d, num_z))

        for array in [p_z, p_w_z, p_d_z]:
            for line in (_.strip() for _ in fd):
                if line == "":
                    break
                else:
                    indexes, prob = line.split("\t")
                    prob = float(prob)
                    xs = [int(i) for i in indexes.split()]
                    if len(xs) == 1:
                        z = xs[0]
                        array[z] = prob
                    else:
                        x, z = xs
                        array[x][z] = prob

        # define instance variables
        self.p_z = p_z
        self.p_w_z = p_w_z
        self.p_d_z = p_d_z

    def p_dwz(self, d, w, z):
        return self.p_z[z] * self.p_w_z[w][z] * self.p_d_z[d][z]

    def dw2vec(self, d, w):
        prob_z = np.zeros(self.num_z)
        for z in range(self.num_z):
            prob_z[z] = self.p_dwz(d, w, z) / \
                sum(self.p_dwz(d, w, _z) for _z in range(self.num_z))
        return prob_z

    def dw2z(self, d, w):
        return np.argmax(self.dw2vec(d, w))

    def d2vec(self, d):
        prob_z = np.zeros(self.num_z)
        for z in range(self.num_z):
            prob_z[z] = sum(self.p_dwz(d, _w, z) for _w in range(self.num_w)) / \
                sum(self.p_dwz(d, _w, _z) for _z in range(self.num_z)
                    for _w in range(self.num_w))
        return prob_z

    def d2z(self, d):
        return np.argmax(self.d2vec(d))

def train(
    corpus,
    num_z: int,
    loop_count: int,
    logger=None
):

    if not logger:
        logger = getLogger(__file__)

    # check the size of docutments and words
    num_d = 0
    num_data = 0
    ws = set()
    for line in corpus:
        num_d += 1
        for w in line:
            num_data += 1
            if w not in ws:
                ws.add(w)
    num_w = len(ws)
    logger.info("num_d: {}, num_w: {}, num_z: {}, num_data: {}".format(
        num_d, num_w, num_z, num_data
    ))

    p_z_dw = np.zeros((num_z, num_d, num_w))
    p_w_z = np.zeros((num_w, num_z))
    p_d_z = np.zeros((num_d, num_z))
    p_z = np.zeros(num_z)

    # initialize parameter `p_z_dw` by dirichlet distribution
    # with parameters alphas == 1
    for i in range(num_d):
        for j in range(num_w):
            params = np.random.dirichlet(np.ones(num_z), 1)[0]
            for k in range(num_z):
                p_z_dw[k][i][j] = params[k]

    for i in range(loop_count):
        # M step
        for z in range(num_z):
            denom = sum(
                p_z_dw[z][d][w] for (d, w) in data(corpus)
            )
            # update P(z), P(w|z), P(d|z)
            p_z[z] = denom / num_data
            for w in range(num_w):
                p_w_z[w][z] = sum(
                    p_z_dw[z][d][w] for d, _w in data(corpus)
                    if w == _w
                ) / denom
            for d in range(num_d):
                p_d_z[d][z] = sum(
                    p_z_dw[z][d][w] for _d, w in data(corpus)
                    if d == _d
                ) / denom

        # E step
        for d in range(num_d):
            for w in range(num_w):
                denom = sum(
                    p_z[z] * p_d_z[d][z] * p_w_z[w][z] for z in range(num_z)
                )
                for z in range(num_z):
                    p_z_dw[z][d][w] = p_z[z] * p_d_z[d][z] * p_w_z[w][z] / denom

    logger.debug("sum of P(z): {}".format(p_z.sum()))
    logger.debug("sum of P(w|z)\n{}".format(p_w_z.T.sum(1)))
    logger.debug("sum of P(d|z)\n{}".format(p_d_z.T.sum(1)))

    # output
    print("{} {} {}".format(num_d, num_w, num_z))
    print("")
    for z in range(num_z):
        print("{}	{:.6f}".format(z, p_z[z]))
    print("")
    for z in range(num_z):
        for w in range(num_w):
            print("{} {}	{:.6f}".format(w, z, p_w_z[w][z]))
    print("")
    for z in range(num_z):
        for d in range(num_d):
            print("{} {}	{:.6f}".format(d, z, p_d_z[d][z]))



if __name__ == "__main__":
    import sys
    filename = sys.argv[1]

    from logging import getLogger, basicConfig, DEBUG, INFO
    import argparse

    # parse arg
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        nargs="?",
        type=str,
        help="filename"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="verbosity option"
    )
    args = parser.parse_args()

    # logger
    logger = getLogger(__name__)
    basicConfig(
        level=DEBUG if args.verbose else INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    dictionary = file2dic(
        line.strip().split() for line in open(args.filename)
    )
    corpus = Corpus(args.filename, dictionary)

    #train(
    #    corpus,
    #    num_z=3,
    #    loop_count=10,
    #    logger=logger
    #)

    plsi = PLSI(corpus, dictionary)
    plsi.load("model")

    for (d, line), words in zip(
            enumerate(corpus),
            (line.strip().split() for line in open(args.filename))):
        res = []
        for w, word in zip(line, words):
            i = plsi.dw2z(d, w)
            res.append("{}:{}".format(word, i))
        print(' '.join(res))
