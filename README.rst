==============================
PLSI library
==============================

Require numpy.

Training
=========

Train first:

.. code-block:: bash

    $ python plsi.py test.txt >model


Example
========

.. code-block:: python

    >>> import plsi

    >>> filename = "test.txt"

    >>> dictionary = plsi.file2dic(
    >>>     line.strip().split() for line in open(filename)
    >>> )
    >>> corpus = plsi.Corpus(filename, dictionary)
    >>> plsi = PLSI(corpus, dictionary)
    >>> plsi.load("model")

    >>> for (d, line), words in zip(
    >>>         enumerate(corpus),
    >>>         (line.strip().split() for line in open(filename))):
    >>>     res = []
    >>>     for w, word in zip(line, words):
    >>>         i = plsi.dw2z(d, w)
    >>>         res.append("{}:{}".format(word, i))
    >>>     print(' '.join(res))
    2015-01-27 22:12:36,954 - plsi.py - INFO - load model: num_d 4, num_w 5, num_z 3
    a:2 b:2 c:0
    a:2 b:2 d:1
    a:2 b:2 d:1
    a:1 c:0 d:1 e:1
