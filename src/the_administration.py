import random
import sys
import time
from dataclasses import dataclass
from typing import Type

import numpy as np
from deap import base, creator, tools

from libs.deap_confidence import ConfidenceFitness, removeDominated
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
        toolbox.register("select", removeDominated)
        hof = tools.HallOfFame(1, similar=np.array_equal)

        NPOP = 10
        MAXIT = 30
        pop = toolbox.population(n=NPOP)

        CXPB, MUTPB = 0.5, 0.2
        NGEN = 200000

        for git in range(NGEN):
            # at least 2 samples for individual
            for ind in pop:
                while len(ind.fitness._samples) < 10:
                    ind.fitness.addSamples([toolbox.sample(ind)])
                # print(f"startup_{g}", ind.fitness.middle, ind.fitness.size)

            maxit = MAXIT
            while True:
                # Try to select
                pop = toolbox.select(pop, int(NPOP/10*3),
                         NPOP if maxit > 0 else int(NPOP/10*8)) # if maxit reached, remove worst
                if len(pop) < NPOP:
                    if (maxit==0):
                        print("By max...")
                    else:
                        print("By DOM!")
                    break # If at least one individual gone, stop.
                # Else, improve interval with best gradient
                pop = sorted(pop, key=lambda ind: ind.fitness.getGradient(5), reverse=True)
                pop[0].fitness.addSamples([toolbox.sample(pop[0])])
                maxit -= 1

                print("## pop lens:", [
                    len(p.fitness._samples) if p.fitness._samples is not None else None for p in pop
                ])

            # pop = sorted(pop, key=lambda ind: ind.fitness.getGradient(3), reverse=True)

            pop = sorted(pop, key=lambda p: p.fitness.middle, reverse=True)
            print("## pop before:", [
                int(p.fitness.middle/1000) if p.fitness.middle is not None else None for p in pop
            ])

            # select best individuals
            # pop = sorted(pop, key=lambda p: p.fitness.middle, reverse=True)
            random.shuffle(pop)
            mutants = pop[:NPOP - len(pop)]
            print("## create frm:", [
                int(p.fitness.middle / 1000) if p.fitness.middle is not None else None for p in mutants
            ])
            offspring = []
            for mutant in mutants:
                clone = toolbox.clone(mutant)
                toolbox.mutate(clone)
                clone.fitness.reset()
                offspring.append(clone)

            for ind in offspring:
                while len(ind.fitness._samples) < 3:
                    ind.fitness.addSamples([toolbox.sample(ind)])

            print("## obtain   :", [
                int(p.fitness.middle / 1000) if p.fitness.middle is not None else None for p in offspring
            ])

            # The population is partially replaced by the offspring
            pop[:] = pop + offspring

            hof.update(pop)
            best = hof[0]
            # pop = sorted(pop, key=lambda ind: ind.fitness.getGradient(3), reverse=True)



            pop = sorted(pop, key=lambda p: p.fitness.middle, reverse=True)
            print("## pop after :", [
                int(p.fitness.middle/1000) if p.fitness.middle is not None else None for p in pop
            ])
            print("bestsv", )

            with open('checkpoint.txt', 'a') as f:
                printpa(best, sym, file=f)

        return hof[0]

    return _paOpt(sym)


def printpa(best,sym, file=sys.stdout):
    fen = PaFenotype(sym, best)
    print("fee", fen.fee, file=file)
    print("th", fen.th, file=file)
    print("pnlt", np.int64(fen.pnlt / 1000), file=file)
    print("Rpc", np.int64(fen.Rpc * 100), file=file)
    print("socval middle", best.fitness.middle, file=file)
    print("socval int", best.fitness._interval, file=file)

# example
if __name__ == '__main__':


    # only 4 test
    from problems.a99 import a99 as sym

    random.seed(54)

    sym.doPrecalcs(sym)

    s = time.time()
    best = paOpt(sym)
    printpa(best,sym)
    t = time.time() - s
    print("Time:", t)














