import numpy as np


class UntilEnough:
    def __init__(self, toCheck: [], bound, maxIter):
        self.current = None
        self._bound = bound
        self._maxIter = maxIter
        self._toCheck = rmvDup(toCheck)

    def __iter__(self):
        return self

    def __next__(self):
        self._maxIter -= 1

        if cmpQuality(self._toCheck) >= self._bound or self._maxIter == 0:
            raise StopIteration

        return sorted(self._toCheck,
            key=lambda ind: len(ind.fitness._samples))[:int(np.ceil(len(self._toCheck)/3))]


def addSamples(targetpop, N, mapper, sampler):
    """
        Optimized way to add N sample
        to each individual in targetpop
    """
    samples = mapper(sampler, targetpop * N)
    for ind, ind_samples in zip(targetpop * N, samples):
        ind.fitness.addSamples(ind_samples)


def rmvDup(tocheck):
    """
        Remove duplicate
    """
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
    if tot == 0:
        return 1
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
