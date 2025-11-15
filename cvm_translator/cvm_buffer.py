import random


class CVMBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.p = 1.0
        self.buffer = {}

    def add(self, element):
        if element in self.buffer:
            del self.buffer[element]
        u = random.random()
        if u < self.p:
            if len(self.buffer) < self.capacity:
                self.buffer[element] = u
            else:
                max_key, max_u = max(self.buffer.items(), key=lambda x: x[1])
                if u > max_u:
                    self.p = u
                else:
                    del self.buffer[max_key]
                    self.buffer[element] = u
                    self.p = max_u

    def estimate(self):
        if self.p <= 0:
            return 0.0
        return float(len(self.buffer)) / float(self.p)

    def cores(self):
        return list(self.buffer.keys())

    def threshold(self):
        return self.p

    def size(self):
        return len(self.buffer)

