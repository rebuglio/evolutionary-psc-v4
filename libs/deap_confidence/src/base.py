import random

import numpy as np
from scipy import stats
from collections import deque
from functools import partial


def expUtility(a, c, unity):
    """
        Returns an exponential utility function, since
        https://en.wikipedia.org/wiki/Exponential_utility
    """
    c = c / unity
    if a == 0:
        return c
    return (1 - np.exp(-a * c)) / a


class ConfidenceFitness():
    """
        ConfidenceFitness is a non deterministic measure of quality of a
        solution. The reliability can be improved by adding new samples,
        with addSamples method.

        Fitnesses may be compared using the ``>``, ``<`` operators.
    """

    def __init__(
            self,
            confidence=0.90,
            history=10,
            utilityfn=partial(expUtility, c=0.05, unity=100000)
    ):
        self._utilityfn = utilityfn
        self._confidence = confidence
        self._sizehistory = deque(maxlen=history)

        self._samples = []
        self._utils = []
        self._interval = (-10000000, 10000000)
        self._intsize = self._interval[1] - self._interval[0]

    def addSample(self, sample):
        self.addSamples([sample])

    def addSamples(self, samples: []):
        self._samples += samples
        self._utils += [self._utilityfn(s) for s in samples]

        self._sizehistory.append(self._intsize)
        self._interval = stats.t.interval(
            self._confidence, len(self._utils) - 1,
            loc=np.mean(self._utils), scale=stats.sem(self._utils))
        self._intsize = self._interval[1] - self._interval[0]

    @property
    def potentialy(self):
        """
            Relative indicator showing "how much the interval can be reduced"
            adding new samples
        """
        return np.mean(self._intvars)

    @property
    def valid(self):
        """
            Assess if a fitness is valid or not.
        """
        return len(self._samples) != 0

    def _delValues(self):
        self._samples = []

    def _getValues(self):
        raise Exception("Cant read value, use comparison operators")

    def _setValues(self, value):
        raise Exception("Cant set directly, use addSamples")

    values = property(_getValues, _setValues, _delValues,
                      ("Fitness values"))

    def __gt__(self, other):
        if len(self._samples)<30 and len(self._samples) < len(other._samples):
            return False
        return self._interval[0] > other._interval[1]

    def __le__(self, other):
        if len(other._samples)<30 and len(other._samples) < len(self._samples):
            return False
        return self._interval[1] < other._interval[0]

    def __ne__(self, other):
        return self > other or self < other

    def __eq__(self, other):
        return not self != other


def rmvDup(tocheck):
    random.shuffle(tocheck)
    ok = []
    for x1 in tocheck:
        find = False
        for x2 in ok:
            diff = np.linalg.norm(x1 - x2)
            if diff == 0: # x1.fitness._utils[-1] == x2.fitness._utils[-1]
                find = True
        if find == False:
            ok.append(x1)
    return ok


def cmpQuality(tocheck):
    cmp, tot = 0, 0
    for i in range(1, len(tocheck)):
        for j in range(i):
            if tocheck[i].fitness != tocheck[j].fitness:
                cmp += 1
            tot += 1

    return cmp / tot


def removeBounded(individuals, atleast, atmost):
    """
        Remove confidence interval which maximum
        is lower than other minimum.
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

    totake = len(individuals) - domCount
    if totake < atleast:
        totake = atleast
    elif totake > atmost:
        totake = atmost
    return individuals[:totake]
