import random

import numpy as np
from scipy import stats
from collections import deque
from functools import partial


def expUtility(c):
    """
        Returns an exponential utility function, since
        https://en.wikipedia.org/wiki/Exponential_utility
    """
    return c
    a = 0.015
    return -np.exp(-a * c / 1000000)


class ConfidenceFitness():
    """
        ConfidenceFitness is a non deterministic measure of quality of a
        solution. The reliability can be improved by adding new samples,
        with addSamples method.

        Fitnesses may be compared using the ``>``, ``<`` operators.
    """

    def __init__(
            self,
            confidence = 0.90,
            history = 10,
            utilityfn = expUtility,
    ):
        self._utilityFn = utilityfn
        self._confidence = confidence
        self._sizeHistory = deque(maxlen=history)

        self._samples = []
        self._utils = []
        self._interval = (-10000000, 10000000)
        self._intsize = self._interval[1] - self._interval[0]

    def addSample(self, sample):
        self.addSamples([sample])

    def addSamples(self, samples: []):
        self._samples += samples
        self._utils += samples
        # self._utils += [self._utilityFn(s) for s in samples]
        # print("check", self._samples, self._utils)

        self._sizeHistory.append(self._intsize)
        self._interval = stats.t.interval(
            self._confidence, len(self._utils) - 1,
            loc=np.mean(self._utils), scale=stats.sem(self._utils))
        self._intsize = self._interval[1] - self._interval[0]

    @property
    def valid(self):
        """
            Assess if a fitness is valid or not.
        """
        return len(self._samples) != 0

    def _delValues(self):
        self._samples = []
        self._utils = []

    def _getValues(self):
        raise Exception("Cant read value, use comparison operators")

    def _setValues(self, value):
        raise Exception("Cant set directly, use addSamples")

    values = property(_getValues, _setValues, _delValues,
                      ("Fitness values"))

    def __gt__(self, other):
        if len(self._samples) < 10 or len(other._samples) < 10:
            return False
        if len(self._samples) * 3 < len(other._samples) * 2:
            return False
        return self._interval[0] > other._interval[1]

    def __le__(self, other):
        if len(self._samples) < 10 or len(other._samples) < 10:
            return False
        if len(other._samples) * 3 < len(self._samples) * 2:
            return False
        return self._interval[1] < other._interval[0]

    def __ne__(self, other):
        return self > other or self < other

    def __eq__(self, other):
        return not self != other


