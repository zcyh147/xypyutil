import os.path as osp
import time
import sys

# project
_script_dir = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, repo_root := osp.abspath(f'{_script_dir}/../..'))
import kkpyutil as util


@util.rerun_lock(name='test_rerun_lock', folder=osp.join(util.get_platform_tmp_dir(), '_util'))
def main(n):
    # share lock with exclusive.py
    # locked if run_exclusive is still running
    file = osp.join(util.get_platform_tmp_dir(), '_util', f'reenter_{n}.json')
    util.save_json(file, {'run': n})


if __name__ == '__main__':
    main(sys.argv[1])
