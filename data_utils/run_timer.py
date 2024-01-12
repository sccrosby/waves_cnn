import time

class RunTimer:

    def __init__(self, name=""):
        self.begin = time.time()
        self.name = name
        self.total_time = 0.0

    def start(self):
        self.begin = time.time()

    def get(self):
        runtime = time.time() - self.begin
        self.begin = time.time()
        self.total_time += runtime
        return runtime

    def log(self):
        self.get()

    def stop(self):
        runtime = time.time() - self.begin
        # print("\t{:s} ran in {:s}".format(self.name, str(runtime)))
        self.total_time += runtime
        print("\t%s ran in %.3f ms" % (self.name, runtime * 1000))
        self.begin = time.time()


