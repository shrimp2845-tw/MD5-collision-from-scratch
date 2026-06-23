# SPDX-License-Identifier: MIT
# Copyright (c) 2026 shrimp2845
# This program is a tribute to all the cryptographers who contributed to cracking MD5

import numpy as np
from numba import njit

def get_mask(lm):
    mask = np.uint32(0)
    for i in lm:
        mask |= (np.uint32(1) << np.uint32(i - 1))
    return mask
    
MZ3, MO3 = get_mask([7, 12, 20]), get_mask([])
MZ4, MO4, MF4 = get_mask([7, 24]), get_mask([5, 6, 12, 20, 32]), get_mask([8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23])
MZ5, MO5 = get_mask([5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 27, 29, 30, 31]), get_mask([1, 3, 6, 23, 28, 32])
MZ6, MO6, MF6 = get_mask([3, 6, 8, 9, 10, 15, 24, 28, 32]), get_mask([1, 7, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 26]), get_mask([2, 4, 5, 25, 27, 29, 30, 31])
MZ7, MO7 = get_mask([1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 17, 27, 28, 29, 30, 31, 32]), get_mask([6, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26])
MZ8, MO8 = get_mask([1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 19, 20, 26, 27, 28, 29, 30, 31, 32]), get_mask([7, 9, 11, 17, 21, 24, 25])
MZ9, MO9, MF9 = get_mask([2, 7, 8, 16, 17, 18, 19, 20, 27]), get_mask([1, 3, 4, 5, 6, 9, 10, 11, 12, 14, 21, 25, 26, 28, 29, 30, 31, 32]), get_mask([13])
MZ10, MO10 = get_mask([1, 2, 8, 9, 14, 24, 32]), get_mask([7, 13, 16, 17, 18, 19, 20, 21, 29, 30, 31])
MZ11, MO11, MF11 = get_mask([1, 9, 13, 14, 18, 19, 20, 29, 31, 32]), get_mask([2, 7, 8, 16, 17, 30]), get_mask([15])
MZ12, MO12, MF12 = get_mask([8, 14, 15, 16, 17, 18, 19, 30, 31, 32]), get_mask([9, 13, 20]), get_mask([25, 26])
MZ13, MO13 = get_mask([8, 9, 26, 32]), get_mask([4, 14, 15, 16, 17, 18, 19, 20, 25, 31])
MZ14, MO14 = get_mask([19, 25, 26, 30, 31, 32]), get_mask([4, 8, 9, 14, 15, 16, 17, 18, 20])
MZ15, MO15 = get_mask([4, 9, 15, 21, 23, 26, 32]), get_mask([16, 25, 30, 31])
MZ16, MO16, MFN16 = get_mask([31, 32]), get_mask([30]), get_mask([22])
MZ17, MO17, MF17 = get_mask([18]), get_mask([31]), get_mask([4, 16, 32])
MZ18, MO18, MF18 = get_mask([]), get_mask([18]), get_mask([30, 32])

IV_a = np.uint32(0x67452301)
IV_b = np.uint32(0xefcdab89)
IV_c = np.uint32(0x98badcfe)
IV_d = np.uint32(0x10325476) 
      
@njit('uint32(uint32, uint32)')
def left_rotate(x, n):
    return np.uint32((x << n) | (x >> (np.uint32(32) - n)))

@njit('uint32(uint32, uint32)')
def right_rotate(x, n):
    return np.uint32((x >> n) | (x << (np.uint32(32) - n)))

@njit('uint32(uint32, uint32, uint32)')
def phi1(x, y, z):
    return (x & y) | (~x & z)
    
@njit('uint32(uint32, uint32, uint32)')
def phi2(x, y, z):
    return (x & z) | (y & ~z)
    
@njit('uint32(uint32, uint32, uint32)')
def phi3(x, y, z):
    return x ^ y ^ z

@njit('uint32(uint32, uint32, uint32)')
def phi4(x, y, z):
    return y ^ (x | ~z)
    
@njit('uint64(uint64)')
def rand64(state):
    state ^= state << np.uint64(13)
    state ^= state >> np.uint64(7)
    state ^= state << np.uint64(17)
    return state
 
@njit('uint32[:](uint32[:], uint32[:], uint32)', cache = True)
def block0(m, q, debug):
    """
    1.using single message modification technic to modify original
    message to satisfy sufficient conditions in first 16 steps
    
    2.using multi message modification technic and small search 
    technic to modify original message to satisfy sufficient conditions in 
    17 ~ 20 steps
    
    3.compute remain steps and pray, if any condition in table 4 not
    satisfied go back to 2., posibility to pass is 1/2³⁵ while using 
    real sufficient condition from Liang et al.
    
    note: there is around 20% probability that differantial goes wrong
    in th 19th step (Im not sure if there is still problems in Liang et al.'s conditions 
    or I mess it up some where). it causes the whole differantial to collapes,  
    and it does not effect the probability of the cases to satisfy all the conditions 
    in 21 ~ 64 steps, so there is overall 20% to fail 
    
    m: original message
    q: buffer to store output of every step for future use of multi message modification
    debugs: throw in a debug flag and it will stop right there and return
    """

    # initialize
    
    a, b, c, d = IV_a, IV_b, IV_c, IV_d
    m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15 = m
    
    # step 1
    a = b + left_rotate((a + phi1(b, c, d) + m0 + np.uint32(0xd76aa478)), np.uint32(7))
    q[0] = a
    
    # step 2
    d = a + left_rotate((d + phi1(a, b, c) + m1 + np.uint32(0xe8c7b756)), np.uint32(12))
    q[1] = d
    
    #step 3
    temp = d + left_rotate((c + phi1(d, a, b) + m2 + np.uint32(0x242070db)), np.uint32(17))
    target = (temp | MO3) & ~MZ3
    m2 = right_rotate((target - d), np.uint32(17)) - c - phi1(d, a, b) - np.uint32(0x242070db)
    c = target
    q[2] = c
    
    # step 4
    temp = c + left_rotate((b + phi1(c, d, a) + m3 + np.uint32(0xc1bdceee)), np.uint32(22))
    target = (temp & ~MF4) | (c & MF4)
    target = (target | MO4) & ~MZ4
    m3 = right_rotate((target - c), np.uint32(22)) - b - phi1(c, d, a) - np.uint32(0xc1bdceee)
    b = target
    q[3] = b
    
    # step 5
    temp = b + left_rotate((a + phi1(b, c, d) + m4 + np.uint32(0xf57c0faf)), np.uint32(7))
    target = (temp | MO5) & ~MZ5
    m4 = right_rotate((target - b), np.uint32(7)) - a - phi1(b, c, d) - np.uint32(0xf57c0faf)
    a = target
    q[4] = a

    # step 6
    temp = a + left_rotate((d + phi1(a, b, c) + m5 + np.uint32(0x4787c62a)), np.uint32(12))
    target = (temp & ~MF6) | (a & MF6)
    target = (target | MO6) & ~MZ6
    m5 = right_rotate((target - a), np.uint32(12)) - d - phi1(a, b, c) - np.uint32(0x4787c62a)
    d = target
    q[5] = d

    # step 7
    temp = d + left_rotate((c + phi1(d, a, b) + m6 + np.uint32(0xa8304613)), np.uint32(17))
    target = (temp | MO7) & ~MZ7
    m6 = right_rotate((target - d), np.uint32(17)) - c - phi1(d, a, b) - np.uint32(0xa8304613)
    c = target
    q[6] = c

    # step 8
    temp = c + left_rotate((b + phi1(c, d, a) + m7 + np.uint32(0xfd469501)), np.uint32(22))
    target = (temp | MO8) & ~MZ8
    m7 = right_rotate((target - c), np.uint32(22)) - b - phi1(c, d, a) - np.uint32(0xfd469501)
    b = target
    q[7] = b

    # step 9
    temp = b + left_rotate((a + phi1(b, c, d) + m8 + np.uint32(0x698098d8)), np.uint32(7))
    target = (temp & ~MF9) | (b & MF9)
    target = (target | MO9) & ~MZ9
    m8 = right_rotate((target - b), np.uint32(7)) - a - phi1(b, c, d) - np.uint32(0x698098d8)
    a = target
    q[8] = a
    
    # step 10
    temp = a + left_rotate((d + phi1(a, b, c) + m9 + np.uint32(0x8b44f7af)), np.uint32(12))
    target = (temp | MO10) & ~MZ10
    m9 = right_rotate((target - a), np.uint32(12)) - d - phi1(a, b, c) - np.uint32(0x8b44f7af)
    d = target
    q[9] = d
    
    # step 11
    temp = d + left_rotate((c + phi1(d, a, b) + m10 + np.uint32(0xffff5bb1)), np.uint32(17))
    target = (temp & ~MF11) | (d & MF11)
    target = (target | MO11) & ~MZ11
    m10 = right_rotate((target - d), np.uint32(17)) - c - phi1(d, a, b) - np.uint32(0xffff5bb1)
    c = target
    q[10] = c
    
    # step 12
    temp = c + left_rotate((b + phi1(c, d, a) + m11 + np.uint32(0x895cd7be)), np.uint32(22))
    target = (temp & ~MF12) | (c & MF12)
    target = (target | MO12) & ~MZ12
    m11 = right_rotate((target - c), np.uint32(22)) - b - phi1(c, d, a) - np.uint32(0x895cd7be)
    b = target
    q[11] = b

    # step 13
    temp = b + left_rotate((a + phi1(b, c, d) + m12 + np.uint32(0x6b901122)), np.uint32(7))
    target = (temp | MO13) & ~MZ13
    m12 = right_rotate((target - b), np.uint32(7)) - a - phi1(b, c, d) - np.uint32(0x6b901122)
    a = target
    q[12] = a

    # step 14
    temp = a + left_rotate((d + phi1(a, b, c) + m13 + np.uint32(0xfd987193)), np.uint32(12))
    target = (temp | MO14) & ~MZ14
    m13 = right_rotate((target - a), np.uint32(12)) - d - phi1(a, b, c) - np.uint32(0xfd987193)
    d = target
    q[13] = d

    # step 15
    temp = d + left_rotate((c + phi1(d, a, b) + m14 + np.uint32(0xa679438e)), np.uint32(17))
    target = (temp | MO15) & ~MZ15
    m14 = right_rotate((target - d), np.uint32(17)) - c - phi1(d, a, b) - np.uint32(0xa679438e)
    c = target
    q[14] = c

    # step 16
    temp = c + left_rotate((b + phi1(c, d, a) + m15 + np.uint32(0x49b40821)), np.uint32(22))
    target = (temp & ~MFN16) | ((~c) & MFN16)
    target = (target | MO16) & ~MZ16
    m15 = right_rotate((target - c), np.uint32(22)) - b - phi1(c, d, a) - np.uint32(0x49b40821)
    b = target
    q[15] = b
         
    # initialize (2, 3
    seed = np.uint64(609799) + np.uint64(m0) # guess what this number stands for owo?
    q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15 = q
    
    while True:
        # if previous attempt failed, restart from here
        nm0, nm1, nm2, nm3, nm4, nm5, nm6, nm7, nm8, nm9, nm10, nm11, nm12, nm13, nm14, nm15 = m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
        a1, d1, c1, b1, a2, d2, c2, b2, a3, d3, c3, b3, a4, d4, c4, b4 = q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15
        
        # compute random a5, we will make M0 to satisfy this random a5 later on
        temp = rand64(seed)
        seed = temp
        temp = (temp & ~MF17) | (b4 & MF17)
        a5 = (temp | MO17) & ~MZ17
        
        # step 18
        temp = a5 + left_rotate(d4 + phi2(a5, b4, c4) + m6 + np.uint32(0xc040b340), np.uint32(9))        
        d5 = (temp & ~MF18) | (a5 & MF18)
        d5 = (d5 | MO18) & ~MZ18
        
        # cause 9, 21, 23th bits in c4 is 0, so if we change 9, 21, 23th bits in b4, it will change 18, 30, 32th bits in d5
        b4 = b4 ^ right_rotate((temp ^ d5), np.uint32(9))
        nm15 = right_rotate((b4 - c4), np.uint32(22)) - b3 - phi1(c4, d4, a4) - np.uint32(0x49b40821)
        d5 = a5 + left_rotate(d4 + phi2(a5, b4, c4) + m6 + np.uint32(0xc040b340), np.uint32(9))
        
        # step 19
        found_step19 = False
        b3_base = b3 & ~np.uint32(0x00F00078)
        
        for i in range(0, 256, 127):
            nb3 = b3_base | np.uint32((i & 0x0F) << 3) | np.uint32(((i >> 4) & 0x0F) << 20)
            nm11 = right_rotate((nb3 - c3), np.uint32(22)) - b2 - phi1(c3, d3, a3) - np.uint32(0x895cd7be)
            sigma19 = c4 + phi2(d5, a5, b4) + nm11 + np.uint32(0x265e5a51)
            c5 = d5 + left_rotate(sigma19, np.uint32(14))
            cond1 = (c5 & np.uint32(1 << 17)) == 0
            cond2 = (c5 & np.uint32(1 << 31)) == (d5 & np.uint32(1 << 31))
            cond3 = (sigma19 & np.uint32(0x000000F8)) != np.uint32(0x000000F8)    
            if cond1 and cond2 and cond3:
                b3 = nb3
                nm12 = right_rotate((a4 - b3), np.uint32(7)) - a3 - phi1(b3, c3, d3) - np.uint32(0x6b901122)
                nm13 = right_rotate((d4 - a4), np.uint32(12)) - d3 - phi1(a4, b3, c3) - np.uint32(0xfd987193)
                nm14 = right_rotate((c4 - d4), np.uint32(17)) - c3 - phi1(d4, a4, b3) - np.uint32(0xa679438e)
                nm15 = right_rotate((b4 - c4), np.uint32(22)) - b3 - phi1(c4, d4, a4) - np.uint32(0x49b40821)            
                found_step19 = True
                break
        if not found_step19:
            continue
        
        # step 20
         
        found_step20 = False
        for i in range(1024):  
            temp = rand64(seed)
            seed = temp
            nb5 = (temp & np.uint32(0x7FFFFFFF)) | (c5 & np.uint32(0x80000000))
            sigma20 = right_rotate(nb5 - c5, np.uint32(20))
            if (sigma20 & np.uint32(0xE0000000)) == np.uint32(0):
                continue            
            b5 = nb5
            found_step20 = True
            break          
        if not found_step20:
            continue
          
        nm1 = right_rotate((a5 - b4), np.uint32(5)) - a4 - phi2(b4, c4, d4) - np.uint32(0xf61e2562)
        nm0 = right_rotate((b5 - c5), np.uint32(20)) - b4 - phi2(c5, d5, a5) - np.uint32(0xe9b6c7aa)       
        a1 = IV_b + left_rotate((IV_a + phi1(IV_b, IV_c, IV_d) + nm0 + np.uint32(0xd76aa478)), np.uint32(7))
        d1 = a1 + left_rotate((IV_d + phi1(a1, IV_b, IV_c) + nm1 + np.uint32(0xe8c7b756)), np.uint32(12))
        nm2 = right_rotate((c1 - d1), np.uint32(17)) - IV_c - phi1(d1, a1, IV_b) - np.uint32(0x242070db)
        nm3 = right_rotate((b1 - c1), np.uint32(22)) - IV_b - phi1(c1, d1, a1) - np.uint32(0xc1bdceee)
        nm4 = right_rotate((a2 - b1), np.uint32(7)) - a1 - phi1(b1, c1, d1) - np.uint32(0xf57c0faf)
        nm5 = right_rotate((d2 - a2), np.uint32(12)) - d1 - phi1(a2, b1, c1) - np.uint32(0x4787c62a)      
        
        if debug:
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15] = nm0, nm1, nm2, nm3, nm4, nm5, nm6, nm7, nm8, nm9, nm10, nm11, nm12, nm13, nm14, nm15
            q = np.array([a1, d1, c1, b1, a2, d2, c2, b2, a3, d3, c3, b3, a4, d4, c4, b4, a5, d5, c5, b5], dtype = 'u4')
            return q
        
        # initialize
        a, b, c, d = a5, b5, c5, d5
        
        # pray
        a = b + left_rotate((a + phi2(b, c, d) + nm5 + np.uint32(0xd62f105d)), np.uint32(5))
        if ((a >> 17) & 1) != ((b >> 17) & 1): continue
        
        d = a + left_rotate((d + phi2(a, b, c) + nm10 + np.uint32(0x02441453)), np.uint32(9))
        if ((d >> 31) & 1) != ((a >> 31) & 1): continue
        if ((a >> 31) & 1) != ((b >> 31) & 1): continue
        
        c = d + left_rotate((c + phi2(d, a, b) + nm15 + np.uint32(0xd8a1e681)), np.uint32(14))
        if ((c >> 31) & 1) != 0: continue
        
        sigma23 = right_rotate(c - d, np.uint32(14))
        if ((sigma23 >> 17) & 1) != 0: continue
        
        b = c + left_rotate((b + phi2(c, d, a) + nm4 + np.uint32(0xe7d3fbc8)), np.uint32(20))
        if ((b >> 31) & 1) != 1: continue
        
        a = b + left_rotate((a + phi2(b, c, d) + nm9 + np.uint32(0x21e1cde6)), np.uint32(5))
        d = a + left_rotate((d + phi2(a, b, c) + nm14 + np.uint32(0xc33707d6)), np.uint32(9))
        c = d + left_rotate((c + phi2(d, a, b) + nm3 + np.uint32(0xf4d50d87)), np.uint32(14))
        b = c + left_rotate((b + phi2(c, d, a) + nm8 + np.uint32(0x455a14ed)), np.uint32(20))

        a = b + left_rotate((a + phi2(b, c, d) + nm13 + np.uint32(0xa9e3e905)), np.uint32(5))
        d = a + left_rotate((d + phi2(a, b, c) + nm2 + np.uint32(0xfcefa3f8)), np.uint32(9))
        c = d + left_rotate((c + phi2(d, a, b) + nm7 + np.uint32(0x676f02d9)), np.uint32(14))
        b = c + left_rotate((b + phi2(c, d, a) + nm12 + np.uint32(0x8d2a4c8a)), np.uint32(20))

        a = b + left_rotate((a + phi3(b, c, d) + nm5 + np.uint32(0xfffa3942)), np.uint32(4))
        d = a + left_rotate((d + phi3(a, b, c) + nm8 + np.uint32(0x8771f681)), np.uint32(11))      
        c = d + left_rotate((c + phi3(d, a, b) + nm11 + np.uint32(0x6d9d6122)), np.uint32(16))
        
        sigma35 = right_rotate(c - d, np.uint32(16))
        if ((sigma35 >> 15) & 1) != 0: continue
        
        b = c + left_rotate((b + phi3(c, d, a) + nm14 + np.uint32(0xfde5380c)), np.uint32(23))
        a = b + left_rotate((a + phi3(b, c, d) + nm1 + np.uint32(0xa4beea44)), np.uint32(4))
        d = a + left_rotate((d + phi3(a, b, c) + nm4 + np.uint32(0x4bdecfa9)), np.uint32(11))
        c = d + left_rotate((c + phi3(d, a, b) + nm7 + np.uint32(0xf6bb4b60)), np.uint32(16))
        b = c + left_rotate((b + phi3(c, d, a) + nm10 + np.uint32(0xbebfbc70)), np.uint32(23))
        a = b + left_rotate((a + phi3(b, c, d) + nm13 + np.uint32(0x289b7ec6)), np.uint32(4))
        d = a + left_rotate((d + phi3(a, b, c) + nm0 + np.uint32(0xeaa127fa)), np.uint32(11))
        c = d + left_rotate((c + phi3(d, a, b) + nm3 + np.uint32(0xd4ef3085)), np.uint32(16))
        b = c + left_rotate((b + phi3(c, d, a) + nm6 + np.uint32(0x04881d05)), np.uint32(23))
        a = b + left_rotate((a + phi3(b, c, d) + nm9 + np.uint32(0xd9d4d039)), np.uint32(4))
        d = a + left_rotate((d + phi3(a, b, c) + nm12 + np.uint32(0xe6db99e5)), np.uint32(11))
        c = d + left_rotate((c + phi3(d, a, b) + nm15 + np.uint32(0x1fa27cf8)), np.uint32(16))
        b = c + left_rotate((b + phi3(c, d, a) + nm2 + np.uint32(0xc4ac5665)), np.uint32(23))
        if ((b >> 31) & 1) != ((d >> 31) & 1): continue
        
        a = b + left_rotate((a + phi4(b, c, d) + nm0 + np.uint32(0xf4292244)), np.uint32(6))
        if ((a >> 31) & 1) != ((c >> 31) & 1): continue
        
        d = a + left_rotate((d + phi4(a, b, c) + nm7 + np.uint32(0x432aff97)), np.uint32(10))
        if ((d >> 31) & 1) != (((b >> 31) & 1) ^ 1): continue
        
        c = d + left_rotate((c + phi4(d, a, b) + nm14 + np.uint32(0xab9423a7)), np.uint32(15))
        if ((c >> 31) & 1) != ((a >> 31) & 1): continue
        
        b = c + left_rotate((b + phi4(c, d, a) + nm5 + np.uint32(0xfc93a039)), np.uint32(21))
        if ((b >> 31) & 1) != ((d >> 31) & 1): continue
        
        a = b + left_rotate((a + phi4(b, c, d) + nm12 + np.uint32(0x655b59c3)), np.uint32(6))
        if ((a >> 31) & 1) != ((c >> 31) & 1): continue
        
        d = a + left_rotate((d + phi4(a, b, c) + nm3 + np.uint32(0x8f0ccc92)), np.uint32(10))
        if ((d >> 31) & 1) != ((b >> 31) & 1): continue
        
        c = d + left_rotate((c + phi4(d, a, b) + nm10 + np.uint32(0xffeff47d)), np.uint32(15))
        if ((c >> 31) & 1) != ((a >> 31) & 1): continue
        
        b = c + left_rotate((b + phi4(c, d, a) + nm1 + np.uint32(0x85845dd1)), np.uint32(21))
        if ((b >> 31) & 1) != ((d >> 31) & 1): continue
        
        a = b + left_rotate((a + phi4(b, c, d) + nm8 + np.uint32(0x6fa87e4f)), np.uint32(6))
        if ((a >> 31) & 1) != ((c >> 31) & 1): continue
        
        d = a + left_rotate((d + phi4(a, b, c) + nm15 + np.uint32(0xfe2ce6e0)), np.uint32(10))
        if ((d >> 31) & 1) != ((b >> 31) & 1): continue
        
        c = d + left_rotate((c + phi4(d, a, b) + nm6 + np.uint32(0xa3014314)), np.uint32(15))
        if ((c >> 31) & 1) != ((a >> 31) & 1): continue
        
        b = c + left_rotate((b + phi4(c, d, a) + nm13 + np.uint32(0x4e0811a1)), np.uint32(21))
        if ((b >> 31) & 1) == ((d >> 31) & 1): continue
        if ((b >> 25) & 1) != 0: continue
        
        a = b + left_rotate((a + phi4(b, c, d) + nm4 + np.uint32(0xf7537e82)), np.uint32(6))
        if ((a >> 25) & 1) != 1: continue    
        if ((a >> 31) & 1) != ((c >> 31) & 1): continue
        
        d = a + left_rotate((d + phi4(a, b, c) + nm11 + np.uint32(0xbd3af235)), np.uint32(10))
        if ((d >> 25) & 1) != 0: continue
        if ((d >> 31) & 1) != ((b >> 31) & 1): continue
        
        sigma62 = right_rotate(d - a, np.uint32(10))
        if ((sigma62 >> 15) & 0x7F) == 0x7F: continue
        
        c = d + left_rotate((c + phi4(d, a, b) + nm2 + np.uint32(0x2ad7d2bb)), np.uint32(15))
        if ((c >> 25) & 1) != 0: continue 
        if ((c >> 31) & 1) != ((a >> 31) & 1): continue

        b = c + left_rotate((b + phi4(c, d, a) + nm9 + np.uint32(0xeb86d391)), np.uint32(21))

        aa = a + IV_a
        bb = b + IV_b
        cc = c + IV_c
        dd = d + IV_d
        
        if ((dd >> 25) & 1) != 0: continue
        if ((cc >> 25) & 1) != 1: continue
        if ((cc >> 26) & 1) != 0: continue
        if ((bb >> 25) & 1) != 0: continue
        if ((bb >> 26) & 1) != 0: continue
        if ((bb >> 5) & 1) != 0: continue
        if ((bb >> 31) & 1) != ((cc >> 31) & 1): continue
        if ((cc >> 31) & 1) != ((dd >> 31) & 1): continue
        
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15] = nm0, nm1, nm2, nm3, nm4, nm5, nm6, nm7, nm8, nm9, nm10, nm11, nm12, nm13, nm14, nm15
        return m
    
def main():
    pass

if __name__ == "__main__":
    main()