import random
import socket
import sys
import time
from multiprocessing import Pool
from typing import Type

import numpy as np
from deap import base, creator, tools

from deap_confidence import ConfidenceFitness, cmpQuality
from deap_confidence.src.base import rmvDup
from loss import socVal
from problems.base import EvPSCSym
from the_operator import opOpt, opPrecalc
from the_randomworld import makeWorlds
from utilities import paGenSlice
from utils.deaputils import cxTwoPointCopy4ndArray
from deap import algorithms
from datatype import PaFenotype


def paSamples(gen: np.ndarray, sym: Type[EvPSCSym], n):
    rws = makeWorlds(sym, n)
    socVals = []
    for rw in rws:
        paFen = PaFenotype(sym, gen)
        E = opOpt(sym, paFen, rw)
        socVals.append(socVal(paFen, sym, rw, E))
    return socVals

toolbox = base.Toolbox()

def paSetup(sym: Type[EvPSCSym]):

    # optimizer setup

    creator.create("ConfidenceFitness", ConfidenceFitness)
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



def paOpt(sym):


    toolbox.register("select", tools.selTournament, tournsize=3)
    # hof = tools.HallOfFame(2, similar=np.array_equal)
    hof = []

    if socket.gethostname() == 'soana':
        POP = 4
        NGEN = 10000
        CXPB, MUTPB = 0.0, 0.4
        MINSMPL = 10
        MAXITER = 30
    else:
        POP = 3
        NGEN = 10000
        CXPB, MUTPB = 0.05, 0.5
        MINSMPL = 2
        MAXITER = 10

    # toolbox.register("bigsample", paSamples, sym=sym, n=MINSMPL)
    toolbox.register("sample", paSamples, sym=sym, n=1)

    pop = toolbox.population(n=POP)

    print("sample")
    pool = Pool()
    toolbox.register("map", pool.map)

    def addSamples(targetpop, N):
        samples = toolbox.map(toolbox.sample, targetpop * N)
        print(len(samples))
        print(len(targetpop*N))
        for ind, ind_samples in zip(targetpop * N, samples):
            ind.fitness.addSamples(ind_samples)

    addSamples(pop, MINSMPL)

    for gen in range(NGEN):
        print("gen:", gen)

        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        print("Need total resample:", len(invalid_ind))
        addSamples(invalid_ind, MINSMPL)

        # Check % of incomparable
        tocheck = rmvDup(pop + hof)
        i = 0
        cmpqlty = cmpQuality(tocheck)
        print("cmpqlty", cmpqlty, "%")
        while cmpqlty < 0.25 and i < MAXITER and len(tocheck) > 1:
            cmpqlty = cmpQuality(tocheck)
            print("cmpqlty", cmpqlty, "%")
            for ind in tocheck:
                print("ind", ind.fitness._interval[0], ind.fitness._interval[1])

            addSamples(tocheck, 5)

            print("grow n", len(tocheck))
            i += 1


        # Pop is replaced
        pop[:] = tools.selTournament(rmvDup(pop + offspring), k=POP, tournsize=3)

        # Hof
        hof[:] = tools.selBest(rmvDup(pop + hof), k=3)

        # Stuff
        print('utils:', [np.mean(h.fitness._utils) for h in hof])
        print("besthof sample, utils:", np.mean(hof[0].fitness._samples), np.mean(hof[0].fitness._utils))
        for i in range(min(3, len(hof))):
            with open(f'checkpoints/hof_{i}.txt', 'a') as f:
                printpa(i, hof[i], sym, file=f)
                f.flush()

    return hof[0]



def printpa(index, best, sym, file=sys.stdout):
    fen = PaFenotype(sym, best)
    print("index", index, file=file)
    print("fee", fen.fee, file=file)
    print("th", fen.th, file=file)
    print("pnlt", np.int64(fen.pnlt / 1000), file=file)
    print("Rpc", np.int64(fen.Rpc * 100), file=file)
    print("samples mean", np.mean(best.fitness._samples), file=file)
    print("samples std", np.std(best.fitness._samples), file=file)
    print("samples n", len(best.fitness._samples), file=file)
    # print("socval int", best.fitness._interval, file=file)



from problems.a99 import a99 as sym

random.seed(54)
np.random.seed(55)

sym.doPrecalcs(sym)
opPrecalc(sym)
paSetup(sym)

if __name__ == '__main__':


    s = time.time()
    best = paOpt(sym)
    printpa(best, sym)
    t = time.time() - s
    print("Time:", t)
