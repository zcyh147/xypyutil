"""
this is src-file docstring
"""


def main():
    """
    this is main docstring
    """
    my_func(100, 0.5, s='bar')
    obj = MyClass()
    obj.my_method(99, 0.99, s='BAR')


class Caller:
    def caller_method(self):
        my_func(-100,
                0.5,
                s='bar')
        obj = MyClass()
        obj.my_method(99, 0.99, s='BAR')


def my_func(i, f, s='foo'):
    pass


class MyClass:
    """
    this is class docstring
    """
    def __init__(self):
        self.i = 99
        self.f = 0.90
        self.s = 'FOO'

    def my_method(self, i, f, s='foo'):
        pass


if __name__ == '__main__':
    main()
