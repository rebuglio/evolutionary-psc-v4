from dataclasses import dataclass
from typing import Type

import numpy as np

from problems.base import EvPSCSym
from utilities import paGenSlice


@dataclass
class PaFenotype():
    def __init__(self, sym: Type[EvPSCSym], gen: np.ndarray):
        self.pnlt = np.zeros((sym.K, sym.T))
        self.th = np.zeros((sym.K, sym.T))

        sls = paGenSlice(sym)

        self.fee = gen[0]
        #print("g:",gen[sls[0]:sls[1]])
        for t in range(sym.T):
            self.pnlt[:,t] = gen[sls[0]:sls[1]]
            self.th[:,t] = gen[sls[1]:sls[2]]
        #print("ch", self.pnlt)
        self.Rpc = np.reshape(gen[sls[2]:sls[3]], (sym.R, 1))


    pnlt: np.ndarray
    th: np.ndarray
    Rpc: np.ndarray
    fee: int