# SPDX-License-Identifier: MIT
# Copyright (c) 2026 shrimp2845
# This program is a tribute to all the cryptographers who contributed to cracking MD5

'''
7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21

0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
'''

import numpy as np
from numba import njit
from block0 import phi1, phi2, phi3, phi4, right_rotate, left_rotate, rand64, get_mask

T16 = np.array(list(map(np.uint32, [0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821])), dtype = 'u4')
R16 = np.array(list(map(np.uint32, [7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22])), dtype = 'u4')

# todo: finish all mask
MZ1, MO1, MF1, MFN1 = get_mask([6, 12, 26, 28]), get_mask([22, 27]), get_mask([]), get_mask([32])

MASKS = np.array([[MZ1, MO1, MF1, MFN1]], dtype = 'u4')

@njit('UniTuple(uint32, 2)(uint32, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uint32)')
def single_mod(s1, s2, s3, s4, mz, mo, mf, mfn, m, t, r):
    temp = s2 + left_rotate(s1 + phi1(s2, s3, s4) + m + t, r)
    target = (temp & ~mfn) | ((~s2) & mfn)
    target = (target & ~mf) | (s2 & mf)
    target = (target | mo) & ~mz
    nm = right_rotate((target - s2), r) - s1 - phi1(s2, s3, s4) - t
    return target, nm
           
@njit('uint32[:](uint32[:], uint32[:], uint32[:])', cache = True)
def block1(m, q, ivs):
    """
    block1 only take 14 random M1, and generate last 2 by it self
    """
    
    # initialize
    aa, bb, cc, dd = ivs
    s1, s2, s3, s4 = aa, bb, cc, dd
    
    """
    because the author is stupid and didn't realize that first 16 step only execute
    once in finding one block until finish first 16 steps of block0, so I'll do it 
    smarter this time
    """
    
    for i in range(14):
        t, r = T16[i], R16[i]        
        mz, mo, mf, mfn = MASKS[i]
        mi = m[i]
        ns, nm = single_mod(s1, s2, s3, s4, mz, mo, mf, mfn, mi, t, r)
        s1, m[i] = ns, nm
        q[i] = s1
        s1, s2, s3, s4 = s4, s1, s2, s3