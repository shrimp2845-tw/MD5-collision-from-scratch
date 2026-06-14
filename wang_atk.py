import numpy as np
from numba import njit

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

A, B, C, D = np.uint32(0x67452301), np.uint32(0xefcdab89), np.uint32(0x98badcfe), np.uint32(0x10325476)

def get_mask(lm):
    mask = np.uint32(0)
    for i in lm:
        mask |= (np.uint32(1) << np.uint32(32 - i))
    return mask
    
MZ3, MO3 = get_mask([7, 12, 20]), get_mask([])
MZ4, MO4, MF4 = get_mask([7, 24]), get_mask([12, 20, 32]), get_mask([8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23])
MZ5, MO5 = get_mask([1, 2, 3, 4, 6, 8, 9, 13, 27, 29, 30, 21, 32]), get_mask([5, 7, 10, 11, 12, 17, 18, 19, 20, 28, 22, 31])
MZ6, MO6, MF6 = get_mask([3, 6, 8, 9, 10, 15, 24, 28, 32]), get_mask([1, 7, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 26]), get_mask([2, 4, 5, 25, 27, 29, 30, 31])
MZ7, MO7 = get_mask([1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 17, 27, 28, 29, 30, 31, 32]), get_mask([6, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26])
MZ8, MO8 = get_mask([1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 19, 20, 26, 27, 28, 29, 30, 31, 32]), get_mask([7, 9, 17, 21, 24, 25])
MZ9, MO9, MF9 = get_mask([2, 7, 8, 16, 17, 18, 19, 20, 27]), get_mask([1, 3, 4, 5, 6, 9, 10, 11, 12, 14, 21, 25, 26, 28, 29, 30, 31, 32]), get_mask([13])
MZ10, MO10 = get_mask([1, 2, 8, 9, 14, 24, 32]), get_mask([7, 13, 16, 17, 18, 19, 20, 21, 31])
MZ11, MO11, MF11 = get_mask([1, 9, 13, 14, 18, 19, 20, 31, 32]), get_mask([2, 7, 8, 17, 16]), get_mask([15])
MZ12, MO12, MF12 = get_mask([8, 14, 15, 16, 17, 18, 19, 31, 32]), get_mask([9, 13, 20]), get_mask([25, 26])
MZ13, MO13 = get_mask([8, 9, 26, 32]), get_mask([4, 14, 15, 16, 17, 18, 20, 25, 31, 19])
MZ14, MO14 = get_mask([4, 26, 32]), get_mask([16, 25, 30])
MZ15, MO15 = get_mask([19, 25, 30, 32]), get_mask([4, 8, 9, 14, 15, 16, 17, 18, 20])
MZ16, MO16 = get_mask([32]), get_mask([30])

@njit('uint32(uint32, uint32)')
def left_rotate(x, n):
    return (x << n) | (x >> (np.uint32(32) - n))

@njit('uint32(uint32, uint32)')
def right_rotate(x, n):
    return (x >> n) | (x << (np.uint(32) - n))

@njit('uint32(uint32, uint32, uint32)')
def phi1(x, y, z):
    return (x & y) | (~x & z)

@njit('void(uint32[:], uint32[:], uint32, uint32, uint32, uint32)')
def phase1(m, q, a, b, c, d):
    """
    using single message modification technic to modify original
    message to satisfy sufficient conditions in first 16 steps

    barely using loops, conditional expressions and arrays for better performance

    need around 2m try/sec

    m: original message
    q: buffer to store output of every step for future use of multi message modification
    """

    # initialize
    m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15 = m
    
    # step 1
    a = b + left_rotate((a + phi1(b, c, d) + m0 + np.uint32(0xd76aa478)), np.uint32(7))
    q[0] = a
    
    # step 2
    d = a + left_rotate((d + phi1(a, b, c) + m1 + np.uint32(0xe8c7b756)), np.uint32(12))
    q[1] = d
    
    #step 3
    temp = d + left_rotate((c + phi1(d, a, b) + m2 + np.uint32(0x242070db)), np.uint32(17))
    c = (temp | MO3) & ~MZ3
    m2 = right_rotate((c - temp), np.uint32(17)) + m2
    q[2] = c
    
    # step 4
    temp = c + left_rotate((b + phi1(c, d, a) + m3 + np.uint32(0xc1bdceee)), np.uint32(22))
    b = (temp & ~MF4) | (c & MF4)
    b = (b | MO4) & ~MZ4
    m3 = right_rotate((b - temp), np.uint32(22)) + m3
    q[3] = b
    
    # step 5
    temp = b + left_rotate((a + phi1(b, c, d) + m4 + np.uint32(0xf57c0faf)), np.uint32(7))
    a = (temp | MO5) & ~MZ5
    m4 = right_rotate((a - temp), np.uint32(7)) + m4
    q[4] = a

    # step 6
    temp = a + left_rotate((d + phi1(a, b, c) + m5 + np.uint32(0x4787c62a)), np.uint32(12))
    d = (temp & ~MF6) | (a & MF6)
    d = (d | MO6) & ~MZ6
    m5 = right_rotate((d - temp), np.uint32(12)) + m5
    q[5] = d

    # step 7
    temp = d + left_rotate((c + phi1(d, a, b) + m6 + np.uint32(0xa8304613)), np.uint32(17))
    c = (temp | MO7) & ~MZ7
    m6 = right_rotate((c - temp), np.uint32(17)) + m6
    q[6] = c

    # step 8
    temp = c + left_rotate((b + phi1(c, d, a) + m7 + np.uint32(0xfd469501)), np.uint32(22))
    b = (temp | MO8) & ~MZ8
    m7 = right_rotate((b - temp), np.uint32(22)) + m7
    q[7] = b

    # step 9
    temp = b + left_rotate((a + phi1(b, c, d) + m8 + np.uint32(0x698098d8)), np.uint32(7))
    a = (temp & ~MF9) | (b & MF9)
    a = (a | MO9) & ~MZ9
    m8 = right_rotate((a - temp), np.uint32(7)) + m8
    q[8] = a
    
    # step 10
    temp = a + left_rotate((d + phi1(a, b, c) + m9 + np.uint32(0x8b44f7af)), np.uint32(12))
    d = (temp | MO10) & ~MZ10
    m9 = right_rotate((d - temp), np.uint32(12)) + m9
    q[9] = d
    
    # step 11
    temp = d + left_rotate((c + phi1(d, a, b) + m10 + np.uint32(0xffff5bb1)), np.uint32(17))
    c = (temp & ~MF11) | (d & MF11)
    c = (c | MO11) & ~MZ11
    m10 = right_rotate((c - temp), np.uint32(17)) + m10
    q[10] = c
    
    # step 12
    temp = c + left_rotate((b + phi1(c, d, a) + m11 + np.uint32(0x895cd7be)), np.uint32(22))
    b = (temp & ~MF12) | (c & MF12)
    b = (b | MO12) & ~MZ12
    m11 = right_rotate((b - temp), np.uint32(22)) + m11
    q[11] = b

    # step 13
    temp = b + left_rotate((a + phi1(b, c, d) + m12 + np.uint32(0x6b901122)), np.uint32(7))
    a = (temp | MO13) & ~MZ13
    m12 = right_rotate((a - temp), np.uint32(7)) + m12
    q[12] = a

    # step 14
    temp = a + left_rotate((d + phi1(a, b, c) + m13 + np.uint32(0xfd987193)), np.uint32(12))
    d = (temp | MO14) & ~MZ14
    m13 = right_rotate((d - temp), np.uint32(12)) + m13
    q[13] = d

    # step 15
    temp = d + left_rotate((c + phi1(d, a, b) + m14 + np.uint32(0xa679438e)), np.uint32(17))
    c = (temp | MO15) & ~MZ15
    m14 = right_rotate((c - temp), np.uint32(17)) + m14
    q[14] = c

    # step 16
    temp = c + left_rotate((b + phi1(c, d, a) + m15 + np.uint32(0x49b40821)), np.uint32(22))
    b = (b | MO16) & ~MZ16
    m15 = right_rotate((b - temp), np.uint32(22)) + m15
    q[15] = b

    m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15] = m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15

def main():
    pass

if __name__ == "__main__":
    main()