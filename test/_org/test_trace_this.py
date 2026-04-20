import sys
import os.path as osp

_script_dir = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.abspath(f'{_script_dir}/../..'))
import xypyutil as util


def hello(n, s, f):
    x = f'hello, {n}, {s}, {f}'
    return x


def test_tracer():
    tracer = util.Tracer(exclude_funcname_pattern='stop')
    tracer.start()
    hello(100, 'world', '0.99')
    tracer.stop()


if __name__ == '__main__':
    test_tracer()
