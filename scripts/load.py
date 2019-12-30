import argparse
from bullinger import annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', str)
    args = parser.parse_args()

    va = annotations.VideoAnnotations(args.filename)
    print(va.data)


if __name__ == '__main__':
    main()
