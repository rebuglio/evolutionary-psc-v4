from dataclasses import dataclass
import numpy as np

from problems.base import EvPSCSym
from utilities import ungroup, group, lslice
from deap.benchmarks import himmelblau

rng = np.random.default_rng(43)

L = [1, 1,1,1,1,1]
Tv = sum(L)
T = 6
J = 2  # jobs
R = 1  # risks = n. exo = n. ext
K = 2

# [j][t]
F_raw = np.transpose([
    [1000,0],
    [0, 250],
    [0, 250],
    [0, 250],
    [0, 250],
    [0, 250],
])



# [r] (column vector)
Apc = np.transpose([
    [0.5]
    # [1,1,1,1,1,1,1,1]
])

# externalities
# r, j
# contribution of job j effort to increase/reduce risk r
def fx(E):
    fx = np.zeros((R, T))

    j = 2
    for t in range(1,2): # 2,3
        fx[j,t] = 1000/3*E[0,0]
    return fx

# kpis
# k, j
# contribution of job j to kpis k
fkmat = np.array([
    [0.5, 0.5],
    [0,   1  ]
])

def fk(E):
    return fkmat @ E

ti = 0.0  # inflazione
ts = 0.05  # tasso sconto reale
tsn = (1 + ti) * (1 + ts) - 1  # tasso sconto nominale
FA_i = [1 / pow((1 + ti), y) for y in range(sum(L))]
FA_sn = [1 / pow((1 + tsn), y) for y in range(sum(L))]
att = np.array(group(FA_sn, L, np.sum))

F = F_raw * group(FA_i, L, np.sum)  # flusso raggruppato per periodi

# risk precalc
RPN = 5000 # samples number

Rv = np.full((R, T), None).tolist()
for r in range(R):
    for t in range(T):
        Rv[r][t] = [0]
Rp = np.full((R, T), None).tolist()

@dataclass
class es1(EvPSCSym):
    R, K, J, T = R, K, J, T
    F = F
    fk, fx = fk, fx
    Apc = Apc

    oprange = (-0.2, 0.2) # np.array([(-0.2, 0.2) for x, y in np.ndindex(F.shape) if F[x, y] != 0])

    parange = [
        np.array([100, 2000]),         # fee
        np.full((K,2), [0, 500]),  # pnlt
        np.full((K,2), [-0.2, 0.2]), # th
        np.full((R,1,2), [0,1])             # Rpc
    ]

    att = att
    Rv, Rp = Rv, Rp


if __name__ == '__main__':
    es1.doPrecalcs(es1)
    print(es1.fk(es1.effortMask))



































