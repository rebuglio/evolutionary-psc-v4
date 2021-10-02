import random

def cxTwoPointCopy4ndArray(ind1, ind2):
    size = len(ind1)
    cx1, cx2 = random.randint(1, size), random.randint(1, size - 1)
    if cx2 >= cx1: cx2 += 1
    else: cx1, cx2 = cx2, cx1
    ind1[cx1:cx2], ind2[cx1:cx2] = ind2[cx1:cx2].copy(), ind1[cx1:cx2].copy()
    return ind1, ind2