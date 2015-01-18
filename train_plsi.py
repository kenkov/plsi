#! /usr/bin/env python
# coding:utf-8


if __name__ == '__main__':
    import sys
    from plsi import train

    file_name = sys.argv[1]
    num_z = int(sys.argv[2])
    loop_count = int(sys.argv[3])

    train(
        file_name,
        num_z,
        loop_count
    )
