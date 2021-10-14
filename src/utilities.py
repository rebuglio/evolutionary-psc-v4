from typing import Type

import numpy as np

from problems.base import EvPSCSym

lslice = lambda L, il: (sum(L[:il]),sum(L[:il+1]))
group = lambda V, L, fn: [fn(V[sum(L[:il]):sum(L[:il+1])]) for il in range(len(L))]
ungroup = lambda V, L: np.array(sum([[x]*l for x, l in zip(V, L)], [])) # sum = tricky flatten
saturate = lambda v,l,h: h if v>h else l if v<l else v

def paGenSlice(sym: Type[EvPSCSym]):
    gelens = sym.K
    return np.cumsum([1, gelens, gelens, sym.R])

def opGen2fen(gen, sym: Type[EvPSCSym]):
    E = np.zeros(sym.F.shape)
    E[sym.effortMask] = gen
    return E

def riskPrecalc(R, T, L, F_raw, R_raw, FA_i):
    # risk precalc
    RPN = 500  # samples number
    Rv = np.full((R, T), None).tolist()
    for r in range(R):
        for t in range(T):
            Rv[r][t] = []
    Rp = np.full((R, T), None).tolist()
    for t in range(T):
        if L[t] == 1:
            for r in range(R):
                frt = F_raw[R_raw[r][t]['i'], t] / FA_i[slice(*lslice(L, t))][0]
                if frt != 0:
                    Rv[r][t] = [x * frt / 100 for x in R_raw[r][t]['v']]
                    Rp[r][t] = R_raw[r][t]['p']
        else:
            for r in range(R):
                tosum = []
                for ut in range(*lslice(L, t)):
                    frt = F_raw[R_raw[r][ut]['i'], t] / FA_i[ut]
                    if frt != 0:
                        v = [x * frt / 100 for x in R_raw[r][t]['v']]
                        p = R_raw[r][t]['p']
                        tosum.append(np.random.choice(v, p=p, size=RPN))
                if (len(tosum) > 0):
                    Rv[r][t] = sum(tosum)
                    Rp[r][t] = None  # all p=1

    return Rv, Rp