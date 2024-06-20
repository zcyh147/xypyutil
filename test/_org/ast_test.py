"""
this is src-file docstring
"""
import os
import os.path as osp
import sys
from subprocess import Popen
from subprocess import run as my_run


def main():
    """
    this is main docstring
    """
    my_func(100, 0.5, s='bar', cls=Caller, lst=[3, 4], attr=__file__.Caller, unsupported={'x': 100}, dtype=list[str])
    obj = MyClass()
    obj.my_method(99, 0.99, s='BAR')


class Caller:
    def __init__(self):
        self.type = str

    def caller_method(self):
        my_func(-100,
                0.5,
                s='bar')
        obj = MyClass()
        obj.my_method(99, 0.99, s='BAR')


def my_func(i, f, s='foo', cls=Caller, lst=(1, 2), attr=None, unsupported=None, dtype=list[str]):
    pass


class MyClass:
    """
    this is class docstring
    """
    def __init__(self):
        # integer
        self.i: int = 99
        # float
        self.f: float = 0.90
        # string
        # variable
        self.s = 'FOO'
        self.caller: __file__.Caller = None
        self.lstInt: list[int] = [0, 1]
        self.lstFloat = (0.8, 0.9)
        self.proxy: __file__.tStrProxy = 'proxy'
        # None is not ast.Name
        self.noneTyped: None = None
        # without annotation or default
        self.unsupported = var

    def my_method(self, i, f, s='foo'):
        pass


tStrProxy = str
var = 100


class NoCtor:
    # comment
    pass


def local_assign():
    s = 'foo'
    s = 'bar'


# comment line 1
# comment line 2
if __name__ == '__main__':
    # comment line 3
    main()  # inline comment
