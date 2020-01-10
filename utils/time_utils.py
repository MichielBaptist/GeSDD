import time
import numpy as np
import utils.string_utils as stru

def get_timer():
    return timer()

"""A simple utility class for timing and creating summary tables in python."""
class timer:
    def __init__(self):
        self.time = None
        self.times = {}

    def restart(self):
        self.__init__()
        self.start()

    def start(self):
        self.now()

    def now(self):
        self.time = time.time()

    def t(self, name):
        if not name in self.times:
            self.times[name] = [self.cut()]
        else:
            self.times[name].append(self.cut())

    def cut(self):
        diff = time.time() - self.time
        self.now()
        return diff

    def total(self):
        return sum([s for (_,s) in self.sum()])

    def avg(self):
        return [(k, np.mean(v)) for (k, v) in self.times.items()]

    def sum(self):
        return [(k, sum(v)) for (k, v) in self.times.items()]

    def pct(self):
        return [(k, s/self.total()) for (k, s) in self.sum()]

    def summary_str(self, name= "Summary of timer"):
        table_head = [
            ("Section", "Abs", "Avg", "Pct")
        ]
        properties = [
            dict(self.sum()),
            dict(self.avg()),
            dict(self.pct())
        ]
        table_body = [(key,) + tuple([prop[key] for prop in properties]) for key in self.times.keys()]
        table_tail = [("Total", self.total(), "--", "1.0")]

        table = table_head + table_body + table_tail

        return stru.pretty_print_table(table, top_bar = True, bot_bar = True, name = name)

    def summary(self, name= "Summary of timer"):
        print(self.summary_str(name))
