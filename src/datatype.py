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
        self.pnlt[sym.effortMask] = gen[sls[0]:sls[1]]
        self.th[sym.effortMask] = gen[sls[1]:sls[2]]
        self.Rpc = np.reshape(gen[sls[2]:sls[3]], (sym.R, 1))


    pnlt: np.ndarray
    th: np.ndarray
    Rpc: np.ndarray
    fee: int