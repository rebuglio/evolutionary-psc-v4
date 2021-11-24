from dataclasses import dataclass
import numpy as np
from collections.abc import Callable

@dataclass
class EvPSCSym():
    kpimask: np.ndarray
    R: int
    K: int
    J: int
    T: int
    F: np.ndarray
    fk: Callable
    fx: Callable
    Apc: np.ndarray
    att: np.ndarray

    Rv: np.ndarray
    Rp: np.ndarray

    oprange: []
    parange: []

    effortMask = None

    def doPrecalcs(self):
        self.effortMask = np.full(self.F.shape, False)
        self.effortMask[self.F != 0] = True

