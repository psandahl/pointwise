from pointwise import ICP, PointCloud

import argparse
import pathlib
import sys


def main() -> bool:
    parser = argparse.ArgumentParser('python -m pointwise')
    parser.add_argument('-r', '--register',
                        nargs=2, type=pathlib.Path,
                        metavar='DATASET',
                        help='Register the second dataset against the first')
    opts = parser.parse_args()

    for dataset in opts.register:
        if not dataset.is_file() or dataset.suffix not in ('.npy', '.xyz'):
            print(
                f"Error: Input dataset '{dataset}' is not readable or not having expected suffix")
            return False

    return run_icp(opts.register[0], opts.register[1])


def run_icp(reference: pathlib.Path, query: pathlib.Path) -> bool:
    icp = ICP()
    icp.set_point_clouds(reference=PointCloud.from_file(path=reference),
                         query=PointCloud.from_file(path=query))
    icp.run()

    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
