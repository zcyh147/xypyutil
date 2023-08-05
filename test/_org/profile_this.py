class Algo:
    def __init__(self, lst, count):
        self.coll = lst
        self.count = count

    def grow(self):
        for c in range(self.count):
            self.coll.append(c)


def run_profile_target():
    algo = Algo([1, 2, 3], count=100)
    algo.grow()
