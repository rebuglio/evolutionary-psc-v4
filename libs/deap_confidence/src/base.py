from typing import Callable

from deap.tools import HallOfFame, selBest
from scipy import stats
import numpy as np
from operator import eq


def expUtility(a, unity):
    """
        Returns an exponential utility function, since
        https://en.wikipedia.org/wiki/Exponential_utility
    """
    def fn(c):
        c = c / unity
        if a == 0:
            return c
        return (1 - np.exp(-a * c)) / a
    return fn

def popReliability(population):
    return np.array([ind.fitness.reliability for ind in population])

class StochasticFitness():
    """
        ConfidenceFitness is a non deterministic measure of quality of a
        solution. The reliability can be improved by adding new samples,
        with addSamples method.

        Fitnesses may be compared using the ``>``, ``<`` operators.
    """

    def __init__(
        self,
        utilityfn = expUtility(0.05, unity=1000000),
        rlbtythresholds = (30, 0.8),
    ):
        self._samples = []
        self._utils = []
        self._utilityfn = utilityfn
        self._rlbtythresholds = rlbtythresholds
        self._ksreliability = 1

    def addSample(self, sample):
        self.addSamples([sample])

    def addSamples(self, samples: []):
        self._samples += samples
        self._utils += [self._utilityfn(s) for s in samples]
        self._ksreliability = 1

    def _compare(self, other):
        pvalue = stats.ks_2samp(self._utils, other._utils).pvalue
        self._ksreliability *= 1 - pvalue
        other._ksreliability *= 1 - pvalue
        return np.mean(self._utils) - np.mean(other._utils)

    @property
    def valid(self):
        """Assess if a fitness is valid or not."""
        return len(self._samples) != 0

    def _getReliability(self):
        w = len(self._samples) / self._rlbtythresholds[0]
        w = w if w < self._rlbtythresholds[1] else self._rlbtythresholds[1]
        return self._ksreliability * (1 - w) + 1 * w

    def _setReliability(self, value):
        raise Exception("Cant set reliability")

    def _delReliability(self):
        self._ksreliability = 1

    def _delValues(self):
        self._samples = []

    def _getValues(self):
        raise Exception("Cant read value, use comparison operators")

    def _setValues(self, value):
        raise Exception("Cant set directly, use addSamples")

    values = property(_getValues, _setValues, _delValues,
                    ("Fitness values"))

    reliability = property(_getReliability, _setReliability, _delReliability,
                    ("Reliability values"))

    def __gt__(self, other):
        return self._compare(other) > 0

    def __le__(self, other):
        return self._compare(other) < 0

    def __eq__(self, other):
        raise Exception("Cant use equal")

    def __ne__(self, other):
        raise Exception("Cant use nequal")



# class StochasticHallOfFame(HallOfFame):
#
#     def __init__(self, maxsize,
#                  reliability=0.95,
#                  maxIter = 100,
#                  similar=eq):
#         super().__init__(maxsize, similar)
#         self._reliability = reliability
#         self._maxIter = maxIter
#         self._hof = []
#
#     def tryUpdate(self, candidate):
#         return selBest([self._hof + candidate], k=super().maxsize)
#
#     def __getitem__(self, i):
#         return self.hof[i]











