import argparse

from ml_serving.utils import helpers


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test background'
    )
    parser.add_argument(
        '--serving-addr',
        help='Background removal serving gRPC address',
        required=True,
    )
    parser.add_argument(
        '--color',
        help='Path to color file',
    )
    parser.add_argument(
        '--depth',
        help='Path to depth file',
    )
    return parser.parse_args()


def main(args):
    outputs = helpers.predict_grpc({'input': ''}, args.serving_addr)


if __name__ == '__main__':
    main(parse_args())
