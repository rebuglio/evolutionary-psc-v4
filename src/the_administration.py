import random
import sys
import time
from typing import Type

import numpy as np
from deap import base, creator, tools

from deap_confidence import StochasticFitness, StochasticHallOfFame
from deap_confidence.src.base import popReliability
from loss import socVal
from problems.base import EvPSCSym
from the_operator import opOpt
from the_randomworld import makeWorlds
from utilities import paGenSlice
from utils.deaputils import cxTwoPointCopy4ndArray
from deap import algorithms
from datatype import PaFenotype


# def buildPaGenotype(sym: Type[EvPSCSym]):
#     # fee, pnlt, th, Rpc
#     gens = [1, sum(sym.effortMask), sum(sym.effortMask), sym.K]
#     return [ for g in ]

# namespace
def paOpt(sym: Type[EvPSCSym]):
    # optimizer setup

    toolbox = base.Toolbox()
    creator.create("ConfidenceFitness", StochasticFitness)
    creator.create("Individual", np.ndarray, fitness=creator.ConfidenceFitness)

    sls = paGenSlice(sym)
    genbounds = np.ndarray((sls[-1], 2))
    genbounds[0] = sym.parange[0]
    genbounds[sls[0]:sls[1]] = sym.parange[1][sym.effortMask]
    genbounds[sls[1]:sls[2]] = sym.parange[2][sym.effortMask]
    genbounds[sls[2]:sls[3]] = [[0, 1] for i in range(sym.R)]
    lower, upper = [l[0] for l in genbounds], [l[1] for l in genbounds]

    def uniform(low, up, size=None):
        return [random.uniform(a, b) for a, b in zip(low, up)]

    toolbox.register("attr_float", uniform, lower, upper, sls[-1])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", cxTwoPointCopy4ndArray)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=10, low=lower, up=upper, indpb=0.2)
    toolbox.register("evaluate", lambda: (_ for _ in ()).throw(("Cant use evaluate in this simulation")))

    # def middleEval(ind, sym: Type[EvPSCSym]):
    #     ind.fitness.getMiddle
    #
    # toolbox.register("middleEvaluate", sym=sym)

    def paSamples(gen: np.ndarray, sym: Type[EvPSCSym]):
        [rw] = makeWorlds(sym, 1)
        paFen = PaFenotype(sym, gen)
        E = opOpt(sym, paFen, rw)
        return socVal(paFen, sym, rw, E)

    def _paOpt(sym):

        toolbox.register("sample", paSamples, sym=sym)
        toolbox.register("select", tools.selTournament, tournsize=3)
        # hof = StochasticHallOfFame(2, similar=np.array_equal)
        hof = []

        POP = 100
        NGEN = 10000
        CXPB = 0.05
        MUTPB = 0.1
        GROWPB = 0.1
        MAXITER = 100

        pop = toolbox.population(n=POP)
        for ind in pop:
            ind.fitness.addSamples([toolbox.sample(ind)])

        for gen in range(NGEN):
            offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            print("Need total resample:", len(invalid_ind))
            for ind in invalid_ind:
                ind.fitness.addSamples([toolbox.sample(ind)])

            # SELECTION

            i = 0
            while True:

                # Try select (each comparison alter reliability)
                trypop = toolbox.select(pop + offspring, POP)

                # Check reliability
                reliability = popReliability(trypop)
                if np.mean(reliability) > 0.80 or i > MAXITER:
                    # if reliability's good or max-iter reached, stop
                    break

                print("rel:", reliability)

                # Grow samples for PGROW % of solutions with worst reliability
                ngrow = int(np.ceil(len(pop + offspring) * GROWPB))
                for ind in sorted(pop + offspring, key=lambda ind: ind.fitness.reliability)[:ngrow]:
                    ind.fitness.addSamples([toolbox.sample(ind)])

                i += 1

            pop = trypop

            # HOF

            i = 0
            while True:
                tryhof = tools.selBest(hof + pop, k=10)

                # Check reliability
                reliability = popReliability(tryhof)
                if np.mean(reliability) > 0.90 or i > MAXITER:
                    # if reliability's good or max-iter reached, stop
                    break

                # Grow samples for PGROW % of solutions with worst reliability
                ngrow = int(np.ceil(len(tryhof) * GROWPB))
                for ind in sorted(hof + pop, key=lambda ind: ind.fitness.reliability)[:ngrow]:
                    ind.fitness.addSamples([toolbox.sample(ind)])

                i += 1

            print("tryhof:", tryhof)
            hof = sorted(tryhof, key=lambda ind: np.mean(ind.fitness._utils), reverse=True)
            print("hof:", [np.mean(ind.fitness._utils) for ind in hof])
            print("hof0:", np.mean(hof[0].fitness._utils))

            # STUFF

            print("besthof:", np.mean(hof[0].fitness._samples))

            with open('checkpoint.txt', 'a') as f:
                printpa(hof[0], sym, file=f)
                f.flush()

        return hof[0]

    return _paOpt(sym)


def printpa(best, sym, file=sys.stdout):
    fen = PaFenotype(sym, best)
    print("fee", fen.fee, file=file)
    print("th", fen.th, file=file)
    print("pnlt", np.int64(fen.pnlt / 1000), file=file)
    print("Rpc", np.int64(fen.Rpc * 100), file=file)
    print("samples mean", np.mean(best.fitness._samples), file=file)
    print("samples std", np.std(best.fitness._samples), file=file)
    print("samples n", len(best.fitness._samples), file=file)
    # print("socval int", best.fitness._interval, file=file)


# example
if __name__ == '__main__':
    # only 4 test
    from problems.a99 import a99 as sym

    random.seed(54)
    np.random.seed(55)

    sym.doPrecalcs(sym)

    s = time.time()
    best = paOpt(sym)
    printpa(best, sym)
    t = time.time() - s
    print("Time:", t)
