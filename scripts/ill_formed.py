import argparse
import glob
import os.path
import logging

from bullinger import annotations


def find_errors():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='errors.txt')
    args = parser.parse_args()

    filenames = glob.glob('resources/**/*.txt')
    wrong = []
    total = 0
    for filename in filenames:
        basename = os.path.basename(filename)
        if basename.startswith('__'):
            continue

        total += 1
        try:
            va = annotations.VideoAnnotations(filename)
            if va.ill_formed:
                wrong.append(filename)
        except Exception as e:
            wrong.append(filename)

    logging.info('{}/{} ill-formed files.'.format(len(wrong), total))

    with open(args.output, 'w') as fp:
        for filename in wrong:
            fp.write(filename + '\n')
    logging.info('Exported as {}'.format(args.output))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    find_errors()
