import kkpyutil as util


def hello(n, s, f):
    x = f'hello, {n}, {s}, {f}'
    return x


if __name__ == '__main__':
    tracer = util.Tracer(exclude_funcname_pattern='stop')
    tracer.ignore_stdlibs()
    tracer.start()
    hello(100, 'world', '0.99')
    tracer.stop()
