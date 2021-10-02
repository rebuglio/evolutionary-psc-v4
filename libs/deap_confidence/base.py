from typing import Callable

from scipy import stats
import numpy as np

class ConfidenceFitness():

    def __init__(self, confidence=0.95):
        self._confidence = confidence
        self.reset()

    def reset(self):
        self._samples = []
        self._interval = [None, None]
        self._middle = None
        self._gradientHistory = []

    def addSamples(self, samples: []):
        self._samples += map(lambda s: s[0], samples)

        newint = None
        if len(self._samples) >= 2:
            newint = stats.t.interval(self._confidence, len(self._samples) - 1,
                             loc=np.mean(self._samples), scale=stats.sem(self._samples))
            # print(newint)

        if len(self._samples) >= 3:
            psize = self.getSize()
            self._middle = np.mean(samples)
            self._interval = newint
            self._gradientHistory.append((psize - self.getSize()) / psize)
            # print("int", self._interval)

        if len(self._samples) >= 2:
            self._interval = newint

        # print('size', self.getSize(), len(self._samples))

    def getMiddle(self):
        return self._middle

    def getMin(self):
        return self._interval[0]

    def getMax(self):
        return self._interval[1]

    def getSize(self):
        if self.getMin() is None:
            return None
        return self.getMax() - self.getMin()

    # gradient is % of interval size reduction
    # after a new sample. Greater gradient = the interval is getting smaller
    def getGradient(self, n=3):
        if n == 0:
            return [0]
        if n > len(self._gradientHistory):
            n = len(self._gradientHistory)
        return np.mean(self._gradientHistory[-n:])

    def __gt__(self, other):
        if other.middle is None and self.middle is not None:
            return True
        if self.middle is None and other.middle is not None:
            return False
        return self.middle > other.middle

    min = property(getMin)
    max = property(getMax)
    size = property(getSize)
    middle = property(getMiddle)

def removeDominated(individuals, atleast, atmost):
    """ Remove dominated confidence interval from individuals.
        Confidence interval is dominated when another minimun
        is bigger than it maximum.
    """
    if len(individuals) < atleast:
        raise Exception(f"Can't select {atleast} element from {len(individuals)} individuals.")
    if atleast > atmost:
        raise Exception(f"Invalid atleast, atmost couple ({atleast}, {atmost})")

    lbound = max([i.fitness.min for i in individuals if i.fitness.min is not None], default=None)
    if lbound is None:
        return individuals[:min(atmost, len(individuals))]

    domDistances = [i.fitness.max - lbound for i in individuals]
    domDistances, individuals = zip(*sorted(zip(domDistances, individuals), reverse=True))
    individuals = list(individuals)
    domCount = sum([1 for d in domDistances if d <= 0])
    print("domcount", domCount, atleast, atmost)

    totake = len(individuals)-domCount
    if totake < atleast:
        totake = atleast
    elif totake > atmost:
        totake = atmost


    return individuals[:totake]























