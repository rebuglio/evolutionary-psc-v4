from dataclasses import dataclass
from typing import Type
import numpy as np
from problems.base import EvPSCSym

rng = np.random.default_rng(43)


@dataclass
class RandomWorld():
    def __init__(self, eps):
        if eps is None:
            raise Exception("Cant istantiate RandomWorld without eps")
        self.eps = eps

    eps: np.ndarray


def makeWorlds(sym: Type[EvPSCSym], W):
    epss = np.zeros((sym.R, sym.T, W))
    for r in range(sym.R):
        for t in range(sym.T):
            if len(sym.Rv[r][t]) != 0:
                epss[r, t] = rng.choice(sym.Rv[r][t], p=sym.Rp[r][t], size=W)
    return [RandomWorld(epss[:, :, w]) for w in range(W)]
