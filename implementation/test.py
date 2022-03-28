class Test:

    def var(self, x):
        print(x)

    def var2(self, y):
        print(y)


class Run:
    def __init__(self, meth):
        self.method = meth

    def runner(self, call="Default"):
        te = Test
        gggg = self.method

