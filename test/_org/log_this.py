import os.path as osp
import kkpyutil as util


@util.logcall(msg='trace',
              logger=util.build_default_logger(
                  logdir=osp.abspath(f'{osp.dirname(__file__)}/../_gen'),
                  name='util',
                  verbose=True))
def myfunc(n, s, f=1.0):
    x = f'hello, {n}, {s}, {f}'
    return x


if __name__ == '__main__':
    myfunc(100, 'hello', f=0.99)
