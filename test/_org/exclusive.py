import os.path as osp
import time
import sys
# project
_script_dir = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, repo_root := osp.abspath(f'{_script_dir}/../..'))
import xypyutil as util


@util.rerun_lock(name='test_rerun_lock', folder=osp.join(util.get_platform_tmp_dir(), '_util'))
def main(sec):
    # share lock with reenter.py
    time.sleep(int(sec))


if __name__ == '__main__':
    main(sys.argv[1])
