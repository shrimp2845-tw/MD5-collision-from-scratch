import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
import time
import math
from block0 import left_rotate, phi1, block0
from block1 import block1

def get_md5_ivs(m):
    T = [int(4294967296 * abs(math.sin(i))) & 0xFFFFFFFF for i in range(1, 65)]
    S = ([7, 12, 17, 22] * 4 +
         [5, 9, 14, 20] * 4 +
         [4, 11, 16, 23] * 4 +
         [6, 10, 15, 21] * 4)

    a, b, c, d = 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476

    m_int = [int(x) for x in m]

    def F(x, y, z):
        return (x & y) | (~x & z)

    def G(x, y, z):
        return (x & z) | (y & ~z)

    def H(x, y, z):
        return x ^ y ^ z

    def I(x, y, z):
        return y ^ (x | ~z)

    def left_rot(x, n):
        return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF

    for i in range(64):
        if i < 16:
            f = F(b, c, d)
            g = i
        elif i < 32:
            f = G(b, c, d)
            g = (5 * i + 1) % 16
        elif i < 48:
            f = H(b, c, d)
            g = (3 * i + 5) % 16
        else:
            f = I(b, c, d)
            g = (7 * i) % 16

        temp = (a + f + m_int[g] + T[i]) & 0xFFFFFFFF
        a = (b + left_rot(temp, S[i])) & 0xFFFFFFFF

        a, b, c, d = d, a, b, c

    aa = (a + 0x67452301) & 0xFFFFFFFF
    bb = (b + 0xefcdab89) & 0xFFFFFFFF
    cc = (c + 0x98badcfe) & 0xFFFFFFFF
    dd = (d + 0x10325476) & 0xFFFFFFFF

    return np.array([aa, bb, cc, dd], dtype=np.uint32)


def normal_block1_step14(m, q, ivs):
    a, b, c, d = ivs[0], ivs[1], ivs[2], ivs[3]

    T16 = [0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501, 0x698098d8,
           0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193]
    R16 = [7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12]

    for i in range(14):
        temp = a + phi1(b, c, d) + m[i] + np.uint32(T16[i])
        a = b + left_rotate(temp, np.uint32(R16[i]))
        q[i] = a
        a, b, c, d = d, a, b, c


def print_horizontal_bit_matrix(q_array, title_name):
    reg_order = (['a', 'd', 'c', 'b'] * 4)[:len(q_array)]
    print("\n" + "=" * 70)
    print(f" {title_name} ({len(q_array)} rows x 32 columns)")
    print("=" * 70)
    for i in range(len(q_array)):
        bin_str = f"{q_array[i]:032b}"
        formatted_bits = " ".join(bin_str[j:j + 4] for j in range(0, 32, 4))
        reg_name = reg_order[i] if i < len(reg_order) else '?'
        print(f"q[{i:<2}]({reg_name}) | {formatted_bits}")


def main():
    st = time.time()
    print('executing......')
    m0_rand = np.random.randint(0, 4294967295, size=16, dtype=np.uint32)
    q_dummy_0 = np.zeros(16, dtype=np.uint32)

    m0 = block0(m0_rand, q_dummy_0, np.uint32(0))
    ivs_normal = get_md5_ivs(m0)

    print(f"IVs (Normal): a=0x{ivs_normal[0]:08x}, b=0x{ivs_normal[1]:08x}, c=0x{ivs_normal[2]:08x}, d=0x{ivs_normal[3]:08x}")

    delta_m0 = np.zeros(16, dtype=np.uint32)
    delta_m0[4] = np.uint32(1) << np.uint32(31)  # +2^31
    delta_m0[11] = np.uint32(1) << np.uint32(15)  # +2^15
    delta_m0[14] = np.uint32(1) << np.uint32(31)  # +2^31
    m0_prime = m0 + delta_m0

    ivs_prime = get_md5_ivs(m0_prime)
    print(f"IVs (Prime) : a=0x{ivs_prime[0]:08x}, b=0x{ivs_prime[1]:08x}, c=0x{ivs_prime[2]:08x}, d=0x{ivs_prime[3]:08x}")

    with np.errstate(over='ignore', under='ignore'):
        delta_h1_calculated = ivs_prime - ivs_normal

    print("\n" + "-" * 70)
    print("Delta H1 (ivs_prime - ivs_normal):")
    print(f"da = 0x{delta_h1_calculated[0]:08x}")
    print(f"db = 0x{delta_h1_calculated[1]:08x}")
    print(f"dc = 0x{delta_h1_calculated[2]:08x}")
    print(f"dd = 0x{delta_h1_calculated[3]:08x}")
    print("-" * 70 + "\n")

    m1_rand = np.random.randint(0, 4294967295, size=16, dtype=np.uint32)
    m1 = m1_rand.copy()
    q_dummy = np.zeros(14, dtype=np.uint32)
    block1(m1, q_dummy, ivs_normal)
    q_returned = q_dummy.copy()

    q_normal = np.zeros(14, dtype=np.uint32)
    normal_block1_step14(m1, q_normal, ivs_normal)

    is_match = np.array_equal(q_returned, q_normal)
    if not is_match:
        for i in range(14):
            if q_returned[i] != q_normal[i]:
                print(f"Mismatch at step {i}: returned=0x{q_returned[i]:08x}, normal=0x{q_normal[i]:08x}")

    delta_m1 = np.zeros(16, dtype=np.uint32)
    delta_m1[4] = np.uint32(1) << np.uint32(31)  # +2^31
    delta_m1[11] = np.uint32(0xFFFF8000)  # -2^15
    delta_m1[14] = np.uint32(1) << np.uint32(31)  # +2^31
    m1_prime = m1 + delta_m1

    q_prime = np.zeros(14, dtype=np.uint32)
    normal_block1_step14(m1_prime, q_prime, ivs_prime)

    print("\n" + "=" * 70)
    print("Message Blocks")
    print("=" * 70)
    for i in range(16):
        diff_flag = " <--" if m1[i] != m1_prime[i] else ""
        print(f"m[{i:<2}] | m1: 0x{m1[i]:08x} | m1': 0x{m1_prime[i]:08x}{diff_flag}")

    print("\n" + "=" * 70)
    print("differential")
    print("=" * 70)
    reg_order = (['a', 'd', 'c', 'b'] * 4)[:14]
    for i in range(14):
        xor_diff = q_normal[i] ^ q_prime[i]
        with np.errstate(over='ignore', under='ignore'):
            mod_diff = np.uint32(q_prime[i] - q_normal[i])
        print(
            f"q[{i:<2}] | {reg_order[i]:<3} | 0x{q_normal[i]:08x} | 0x{q_prime[i]:08x} | 0x{xor_diff:08x} | 0x{mod_diff:08x}")

    print_horizontal_bit_matrix(q_normal, "bits of q")
    xor_array = q_normal ^ q_prime
    print_horizontal_bit_matrix(xor_array, "xor differential")
    print(f'\n Time: {time.time()-st:.2f} sec')

if __name__ == "__main__":
    main()