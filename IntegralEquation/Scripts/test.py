import numpy as np


def func1(obj, c):
    return obj.a * c


class aclass:

    def __init__(self):
        self._a = 2.0

    @property
    def a(self):
        return self._a

    def func(self, b=2.0):
        b += self._a
        x = func1(obj=self, c=b)

        return x


if __name__ == "__main__":

    cl = aclass()
    print(cl.func(b=5.0))
