
class ImproveIterator:
    def __init__(self, low, high):
        self.current = low - 1
        self.high = high

    def __iter__(self):
        return self

    def __next__(self):  # Python 2: def next(self)
        self.current += 1
        if self.current < self.high:
            return self.current
        raise StopIteration

if __name__ == '__main__':
    for c in ImproveIterator(3, 9):
        print(c)
