#! /usr/bin/env python
# coding:utf-8


if __name__ == "__main__":
    from logging import getLogger, basicConfig, DEBUG, INFO
    import argparse
    import plsi

    # parse arg
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        type=str,
        help="filename"
    )
    parser.add_argument(
        "-n", "--numz",
        nargs="?",
        default=3,
        type=int,
        help="the number of topics"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="show DEBUG log"
    )
    args = parser.parse_args()

    # logger
    logger = getLogger(__name__)
    basicConfig(
        level=DEBUG if args.verbose else INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    dictionary = plsi.file2dic(args.filename)
    corpus = plsi.Corpus(args.filename, dictionary)

    plsi.train(
        corpus,
        num_z=args.numz,
        loop_count=10,
        logger=logger
    )
