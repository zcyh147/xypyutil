import kkpyutil as util


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
