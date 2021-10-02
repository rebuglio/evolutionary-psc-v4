from typing import Type

import numpy as np

from datatype import PaFenotype
from problems.base import EvPSCSym
from the_randomworld import RandomWorld
from utilities import opGen2fen

def socVal(
    pa: PaFenotype,
    sym: Type[EvPSCSym],
    rw: Type[RandomWorld],
    E: np.ndarray
):
    p = pa.pnlt * (sym.fk(E) < pa.th)    # faster then np.where
    e_star = rw.eps + sym.fx(E) * sym.Apc  # rischio spacciato alla pa
    r = np.where(e_star > 0, e_star * (1 - pa.Rpc), e_star)
    rpa = np.where(e_star > 0, e_star * pa.Rpc, 0)

    OPG = np.sum(  # over time
        (
            - np.sum(sym.F * (1 + E), axis=0)
            - np.sum(r, axis=0)
            - np.sum(p, axis=0)
            + np.sum(pa.fee)
        ) * sym.att
    )

    SV = np.sum(
        (
            + np.sum(sym.F * (1 + E), axis=0)
            - np.sum(rpa, axis=0)
            - np.sum(pa.fee)
        ) * sym.att
    )

    if OPG < 0:
        SV = SV + 2 * OPG

    return SV, # dont remove comma


def opLoss(
        gen: np.ndarray,
        sym: Type[EvPSCSym],
        pa: Type[PaFenotype],
        rw: Type[RandomWorld]
):
    E = opGen2fen(gen, sym)
    p = pa.pnlt * (sym.fk(E) < pa.th)  # faster then np.where
    e_star = rw.eps + sym.fx(E) * sym.Apc  # rischio spacciato alla pa
    r = np.where(e_star > 0, e_star * (1 - pa.Rpc), e_star)
    return np.sum(  # over time
        (
                - np.sum(sym.F * (1 + E), axis=0)
                - np.sum(r, axis=0)
                - np.sum(p, axis=0)
                + np.sum(pa.fee)
        ) * sym.att
    ),  # dont remove this comma

