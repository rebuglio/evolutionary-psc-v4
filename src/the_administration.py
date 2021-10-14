import random
import socket
import sys
import time
from copy import deepcopy
from multiprocessing import Pool
from typing import Type

import numpy as np
from deap import base, creator, tools

from deap_confidence import ConfidenceFitness, rmvDup, UntilEnough, addSamples
from deap_confidence.src.utils import cmpQuality
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

def paSetup2(sym: Type[EvPSCSym]):
    print("setup")

    # optimizer setup

    creator.create("ConfidenceFitness", ConfidenceFitness)
    creator.create("Individual", np.ndarray, fitness=creator.ConfidenceFitness)

    sls = paGenSlice(sym)
    genbounds = np.ndarray((sls[-1], 2))
    genbounds[0] = sym.parange[0]
    genbounds[sls[0]:sls[1]] = sym.parange[1] # [np.full((2,6), True)] #[sym.effortMask] # pnlt
    genbounds[sls[1]:sls[2]] = sym.parange[2] # [np.full((2,6), True)] #[sym.effortMask] # th
    genbounds[sls[2]:sls[3]] = [[0, 1] for i in range(sym.R)]
    lower, upper = [l[0] for l in genbounds], [l[1] for l in genbounds]

    print("x",lower)
    print("k",upper)

    def uniform(low, up, size=None):
        return [random.uniform(a, b) for a, b in zip(low, up)]

    toolbox.register("attr_float", uniform, lower, upper, sls[-1])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", cxTwoPointCopy4ndArray)
    toolbox.register("evaluate", lambda: (_ for _ in ()).throw(("Cant use evaluate in this simulation")))

    def mutate(ind, low, up, indpb):
        eta = random.randint(2,12)
        return tools.mutPolynomialBounded(ind, eta, low, up, indpb)



    toolbox.register("mutate", mutate, low=lower, up=upper, indpb=0.2)


def paOpt(sym):

    if socket.gethostname() == 'soana':
        POP = 16
        NGEN = 10000
        CXPB, MUTPB = 0.1, 0.3
        MINSMPL = 10
        MAXITER = 50
    else:
        POP = 2
        NGEN = 10000
        CXPB, MUTPB = 0, 0.5
        MINSMPL = 1
        MAXITER = 10

    pool = Pool()
    toolbox.register("map", pool.map)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("paSample", paSamples, sym=sym, n=1)
    toolbox.register("addSamples", addSamples, mapper=toolbox.map, sampler=toolbox.paSample)

    # Initialize pop with MINSMPL samples
    pop = toolbox.population(n=POP)
    toolbox.addSamples(pop, MINSMPL)
    hof = []

    for gen in range(NGEN):

        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        toolbox.addSamples(invalid_ind, MINSMPL)

        # Check % of comparable
        for target in UntilEnough(pop + hof + offspring, 0.25, MAXITER):
            print("lvl:", cmpQuality(pop + hof + offspring))
            toolbox.addSamples(target, 5)

        # Replace pop
        pop[:] = tools.selTournament(rmvDup(pop + offspring), k=POP, tournsize=3)

        # Update hof
        hof[:] = [deepcopy(ind) for ind in tools.selBest(rmvDup(pop + hof), k=3)]

        # Stuff
        debugPrint(hof)



def debugPrint(hof):
    # print('utils:', [np.mean(h.fitness._utils) for h in hof])
    tp=hof[0]
    print(f"# besthof sample, utils:", np.mean(tp.fitness._samples), np.mean(tp.fitness._utils))
    for i in range(min(3, len(hof))):
        with open(f'checkpoints/hof_{i}.txt', 'a') as f:
            printpa(i, hof[i], sym, file=f)
            f.flush()

def printpa(index, best, sym, file=sys.stdout):
    fen = PaFenotype(sym, best)
    print("index", index, file=file)
    print("fee", fen.fee, file=file)
    print("th", fen.th, file=file)
    print("pnlt", np.int64(fen.pnlt / 1), file=file)
    print("Rpc", np.int64(fen.Rpc * 100), file=file)
    print("samples mean", np.mean(best.fitness._samples), file=file)
    print("samples mean", best.fitness._samples, file=file)
    print("samples std", np.std(best.fitness._samples), file=file)
    print("samples n", len(best.fitness._samples), file=file)
    # print("socval int", best.fitness._interval, file=file)



from problems.es1 import es1 as sym

# 54, 55
random.seed(70)
np.random.seed(70)

sym.doPrecalcs(sym)
opPrecalc(sym)
paSetup2(sym)

if __name__ == '__main__':

    s = time.time()
    paOpt(sym)
    t = time.time() - s
    print("Time:", t)
