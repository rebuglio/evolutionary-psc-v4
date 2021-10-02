import random
import time
from dataclasses import dataclass
from typing import Type

import numpy as np
from deap import base, creator, tools

from deap_confidence import ConfidenceFitness, removeDominated
from loss import socVal
from problems.base import EvPSCSym
from the_operator import opOpt
from the_randomworld import RandomWorld, makeWorlds
from utilities import paGenSlice
from utils.deaputils import cxTwoPointCopy4ndArray
from datatype import PaFenotype


# def buildPaGenotype(sym: Type[EvPSCSym]):
#     # fee, pnlt, th, Rpc
#     gens = [1, sum(sym.effortMask), sum(sym.effortMask), sym.K]
#     return [ for g in ]

# namespace
def paOpt(sym: Type[EvPSCSym]):

    # optimizer setup

    toolbox = base.Toolbox()
    creator.create("FitnessConfInt", ConfidenceFitness)
    creator.create("Individual", np.ndarray, fitness=creator.FitnessConfInt, confidence=0.95)

    sls = paGenSlice(sym)
    genbounds = np.ndarray((sls[-1], 2))
    genbounds[0] = sym.parange[0]
    genbounds[sls[0]:sls[1]] = sym.parange[1][sym.effortMask]
    genbounds[sls[1]:sls[2]] = sym.parange[2][sym.effortMask]
    genbounds[sls[2]:sls[3]] = [[0,1] for i in range(sym.R)]
    lower, upper = [l[0] for l in genbounds], [l[1] for l in genbounds]

    def uniform(low, up, size=None):
        return [random.uniform(a, b) for a, b in zip(low, up)]

    toolbox.register("attr_float", uniform, lower, upper, sls[-1])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", cxTwoPointCopy4ndArray)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=10, low=lower, up=upper, indpb=0.05)
    toolbox.register("evaluate", lambda: (_ for _ in ()).throw(("Cant use evaluate in this simulation")))

    # def middleEval(ind, sym: Type[EvPSCSym]):
    #     ind.fitness.getMiddle
    #
    # toolbox.register("middleEvaluate", sym=sym)

    def paSample(gen: np.ndarray, sym: Type[EvPSCSym]):
        [rw] = makeWorlds(sym, 1)
        paFen = PaFenotype(sym, gen)
        E = opOpt(sym, paFen, rw)
        return socVal(paFen, sym, rw, E)

    def _paOpt(sym):

        toolbox.register("sample", paSample, sym=sym)
        toolbox.register("select", removeDominated)
        hof = tools.HallOfFame(1, similar=np.array_equal)

        NPOP = 2
        MAXIT = 10
        pop = toolbox.population(n=NPOP)

        CXPB, MUTPB = 0.5, 0.2
        NGEN = 5

        for git in range(NGEN):
            # at least 2 samples for individual
            for ind in pop:
                while len(ind.fitness._samples) < 3:
                    ind.fitness.addSamples([toolbox.sample(ind)])
                # print(f"startup_{g}", ind.fitness.middle, ind.fitness.size)

            maxit = MAXIT
            while True:
                print("p0", pop[0].fitness.getSize(), len(pop[0].fitness._samples))
                print("p1", pop[1].fitness.getSize(), len(pop[1].fitness._samples))

                # Try to select
                print(len(pop))
                pop = toolbox.select(pop, NPOP//2,
                         NPOP if maxit > 0 else NPOP - 1) # if maxit reached, remove worst
                print(len(pop))
                if len(pop) < NPOP:
                    break # If at least one individual gone, stop.
                # Else, improve interval with best gradient
                pop = sorted(pop, key=lambda ind: ind.fitness.getGradient(3), reverse=True)
                pop[0].fitness.addSamples([toolbox.sample(pop[0])])
                maxit -= 1

            # select best individuals
            bestinds = pop[:NPOP - len(pop)]

            offspring = []
            for mutant in bestinds:
                clone = toolbox.clone(mutant)
                toolbox.mutate(clone)
                clone.fitness.reset()
                offspring.append(clone)

            pop[:] = pop + offspring

            # The population is entirely replaced by the offspring
            hof.update(pop)

            best = hof[0]
            print("## middlesv:", hof[0].fitness.middle)
            # print(best.fitness.getMiddle, best.fitness.getMin, best.fitness.getMax)


            # print(toolbox.evaluate)

        return hof[0]

    return _paOpt(sym)


# example
if __name__ == '__main__':
    # only 4 test
    from problems.a99 import a99 as sym

    sym.doPrecalcs(sym)

    s = time.time()
    best = paOpt(sym)
    fen = PaFenotype(sym, best)
    print("fee", fen.fee)
    print("th", fen.th)
    print("pnlt", np.int64(fen.pnlt/1000))
    print("Rpc", np.int64(fen.Rpc*100))
    print("socval middle", best.fitness.middle)
    print("socval int", best.fitness._interval)
    t = time.time() - s
    print("Time:", t)














