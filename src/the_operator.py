import multiprocessing
import random
import time
from dataclasses import dataclass
from typing import Type

import numpy as np
from deap import algorithms, base, creator, tools
from scipy import stats as st

from loss import opLoss
from problems.base import EvPSCSym
from datatype import PaFenotype
from the_randomworld import RandomWorld, makeWorlds
from utilities import opGen2fen
from utils.deaputils import cxTwoPointCopy4ndArray

creator.create("OpFitnessMax", base.Fitness, weights=(1.0,))
creator.create("OpIndividual", np.ndarray, fitness=creator.OpFitnessMax)

# namespace
def opOpt(sym: Type[EvPSCSym], pa: PaFenotype, rw: Type[RandomWorld]):

    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.uniform, sym.oprange[0], sym.oprange[1])
    toolbox.register("individual", tools.initRepeat, creator.OpIndividual, toolbox.attr_bool, n=np.sum(sym.effortMask))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", cxTwoPointCopy4ndArray)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=10, low=-0.2, up=0.2, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    toolbox.register("evaluate", opLoss, sym=sym, pa=pa, rw=rw)

    #pool = multiprocessing.Pool()
    #toolbox.register("map", pool.map)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1, similar=np.array_equal)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40,
                        halloffame=hof, verbose=False)

    return opGen2fen(hof[0], sym)



# example
if __name__ == '__main__':
    # only 4 test
    from problems.a99 import a99 as sym

    random.seed(1)

    sym.doPrecalcs(sym)

    rws = makeWorlds(sym, 10)

    #eps = np.zeros((sym.R, sym.T))
    fee = 600000
    pnlt = np.ones((sym.K, sym.T)) * 100000
    th = np.ones((sym.K, sym.T)) * 0.1
    Rpc = np.zeros((sym.R, 1))


    @dataclass
    class pa(PaFenotype):
        fee, pnlt, th, Rpc = fee, pnlt, th, Rpc


    # @dataclass
    # class rw(RandomWorld):
    #     eps = eps




    s = time.time()
    Ebests = [opOpt(sym, pa, rw) for rw in rws]
    print(Ebests)
    print(st.normaltest(Ebests))

    #
    # print(np.int64(sym.F / 1000))
    # print(Ebest)
    # t = time.time() - s
    # print("Time:", t)
