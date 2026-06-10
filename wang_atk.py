# This project is very hard-coded due to optimizing and handling complications in differential

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


def get_mask(list0, list1):
    mask0, mask1 = np.uint32(0), np.uint32(0)
    for i in list0:
        mask0 |= (np.uint32(1) << np.uint32(32 - i))
    for i in list1:
        mask1 |= (np.uint32(1) << np.uint32(32 - i))
    return np.uint32(mask0), np.uint32(mask1)

MZ3, MO3 = get_mask([7, 12, 20], [])
MZ4. MO4 = get_mask([], [])

@njit('uint32(uint32)')
def left_rotate(x, n):
    return (x << n) | (x >> (uint32(32) - n))

@njit('uint32(uint32, uint32, uint32)')
def phi1(x, y, z):
    return (x & y) | (~x & z)

@njit('uint32[:](uint32[:])')
def phase1(m):
    # initialize
    m0 = m
    q = np.zeros(64, dtype = np.uint32)
    a, b, c, d = np.uint32(0x67452301), np.uint32(0xefcdab89), np.uint32(0x98badcfe), np.uint32(0x10325476)

    # step 1
    q[0] = b + left_rotate((a + phi1(b, c, d) + m0[0] + np.uint32(0xd76aa478)), np.uint32(7))
    a = q[0]

    # step 2
    q[1] = a + left_rotate((d + phi1(a, b, c) + m0[1] + np.uint32(0xe8c7b756)), np.uint32(12))
    d = q[1]

    # step 3
    temp = d + left_rotate((c + phi1(d, a, b) + m0[2] + np.uint32(0x242070db)), np.uint32(17))

def main():
    pass

if __name__ == "__main__":
    main()