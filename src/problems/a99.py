from dataclasses import dataclass
import numpy as np

from src.problems.base import EvPSCSym
from src.utilities import ungroup, group, lslice

rng = np.random.default_rng(43)

L = [1, 1, 4, 1, 1, 22]
Tv = sum(L)
T = 6
J = 13  # jobs
R = 8  # risks = n. exo = n. ext
K = 13

# [j][t]
F_raw = np.transpose([
    [34884, 113374, 7412, 229800, 21179, 1491, 0, 0, 0, 0, 0, 0, 0],
    [0, 113374, 7412, 229800, 21179, 1491, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 7412, 229800, 21179, 1491, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 26376 // 24, 266156 // 24, 316517 // 24, 121426 // 24, 105588 // 24],
    [0, 0, 0, 0, 0, 0, 0, 108675 // 23, 26376 // 24, 266156 // 24, 316517 // 24, 121426 // 24, 105588 // 24],
    [0, 0, 0, 0, 0, 0, 508845 // 22, 108675 // 23, 26376 // 24, 266156 // 24, 316517 // 24, 121426 // 24,
     105588 // 24]
])

# [r][t]
R_raw = [
    ungroup([{'v': [0, 0, 20, 60, 80], 'p': [0.00, 0.24, 0.31, 0.25, 0.20], 'i': 0}], [Tv]),  # j0
    ungroup([{'v': [3, 0, 5, 8, 12], 'p': [0.05, 0.20, 0.40, 0.20, 0.15], 'i': 1}], [Tv]),  # j1
    ungroup([{'v': [0, 0, 8, 12, 15], 'p': [0.00, 0.22, 0.20, 0.28, 0.30], 'i': 2}], [Tv]),  # j2
    ungroup([{'v': [-10, 0, 10, 20, 30], 'p': [0.01, 0.10, 0.44, 0.38, 0.07], 'i': 3}], [Tv]),  # j3a
    ungroup([{'v': [-5, 0, 7, 12, 15], 'p': [0.05, 0.20, 0.25, 0.20, 0.30], 'i': 3}], [Tv]),  # j3b
    ungroup([{'v': [0, 0, 2, 7, 10], 'p': [0.00, 0.25, 0.35, 0.25, 0.15], 'i': 5}], [Tv]),  # j5
    ungroup([{'v': [0, 0, 8, 4, 12], 'p': [0.00, 0.30, 0.35, 0.25, 0.10], 'i': 6}], [Tv]),  # j6
    ungroup([
        {'v': [0, 0, 10, 20, 30], 'p': [0.00, 0.35, 0.30, 0.25, 0.10], 'i': 7},  # j7.1
        {'v': [0, 0, 10, 20, 30], 'p': [0.00, 0.35, 0.26, 0.27, 0.12], 'i': 8}  # j7.2
    ], [11, Tv - 11])
]

# [r] (column vector)
Apc = np.transpose([
    [0, 0, 0, 0, 0, 0, 0, 0]
    # [1,1,1,1,1,1,1,1]
])

# externalities
# r, j
# contribution of job j effort to increase/reduce risk r
def fx(E):
    fx = np.zeros((R, T))
    # fx[6, 5] = -E[3, 0]
    return fx

# kpis
# k, j
# contribution of job j to kpis k
# fkmat = np.identity(K)
# fkmat =  [
#     [1,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,1,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,1,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,1,0,0,0]
# ]
fkmat = np.identity(K)

def fk(E):
    return fkmat @ E

ti = 0.025  # inflazione
ts = 0.05  # tasso sconto reale
tsn = (1 + ti) * (1 + ts) - 1  # tasso sconto nominale
FA_i = [1 / pow((1 + ti), y) for y in range(sum(L))]
FA_sn = [1 / pow((1 + tsn), y) for y in range(sum(L))]
att = np.array(group(FA_sn, L, np.sum))

F = F_raw * group(FA_i, L, np.sum)  # flusso raggruppato per periodi

# risk precalc
RPN = 500 # samples number
Rv = np.full((R, T), None).tolist()
for r in range(R):
    for t in range(T):
        Rv[r][t] = []
Rp = np.full((R, T), None).tolist()
for t in range(T):
    if L[t] == 1:
        for r in range(R):
            frt = F_raw[R_raw[r][t]['i'],t] / FA_i[slice(*lslice(L,t))][0]
            if frt != 0:
                Rv[r][t] = [x * frt/100 for x in R_raw[r][t]['v']]
                Rp[r][t] = R_raw[r][t]['p']
    else:
        for r in range(R):
            tosum = []
            for ut in range(*lslice(L,t)):
                frt = F_raw[R_raw[r][ut]['i'],t] / FA_i[ut]
                if frt != 0:
                    v = [x * frt/100 for x in R_raw[r][t]['v']]
                    p = R_raw[r][t]['p']
                    tosum.append(rng.choice(v,p=p,size=RPN))
            if (len(tosum)>0):
                Rv[r][t] = sum(tosum)
                Rp[r][t] = None # all p=1

@dataclass
class a99(EvPSCSym):
    R, K, J, T = R, K, J, T
    F = F
    fk, fx = fk, fx
    Apc = Apc

    oprange = (-0.2, 0.2) # np.array([(-0.2, 0.2) for x, y in np.ndindex(F.shape) if F[x, y] != 0])

    parange = [
        np.array([400000, 800000]),         # fee
        np.dstack((np.zeros(F.shape), F)),  # pnlt
        np.full((*F.shape,2), [-0.2, 0.2]), # th
        np.full((R,1,2), [0,1])             # Rpc
    ]

    att = att
    Rv, Rp = Rv, Rp


if __name__ == '__main__':
    print(a99.parange)










