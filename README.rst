==============================
PLSI library
==============================

必要なもの
===========

*   numpy

実行している環境では `numpy (1.10.0.dev-b69035e)` を使っている。

モデル学習
============

まずはモデルを学習します。

.. code-block:: bash

    $ python train_plsi.py test.txt -n 10 >model
    2015-02-04 05:00:59,605 - __main__ - INFO - num_d: 4, num_w: 5, num_z: 3, num_data: 13

オプションの意味は

.. code-block:: bash

    $ python train_plsi.py -h
    usage: train_plsi.py [-h] [-n [NUMZ]] [-v] filename

    positional arguments:
      filename              filename

    optional arguments:
      -h, --help            show this help message and exit
      -n [NUMZ], --numz [NUMZ]
                            the number of topics
      -v, --verbose         show DEBUG log


トピックを求める
==================

test.txt の各単語のトピックは `plsi.py` を使って求めることができます。

.. code-block:: bash

    $ python plsi.py model test.txt
    2015-02-04 05:02:07,034 - plsi.py - INFO - load model: num_d 4, num_w 5, num_z 3
    a:0 b:0 c:0
    a:1 b:1 d:1
    a:1 b:1 d:1
    a:2 c:2 d:2 e:2

オプションの意味は

.. code-block:: bash

    python plsi.py -h
    usage: plsi.py [-h] [-v] model filename

    positional arguments:
      model          model file
      filename       filename

    optional arguments:
      -h, --help     show this help message and exit
      -v, --verbose  show DEBUG log

です。必ず `model` と `filename` を指定してください。
