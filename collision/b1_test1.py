import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
import time
import math
from block1 import block1


HARDCODED_IVS = [
    {
        "normal": np.array([0xb6468fb3, 0xd117bade, 0x9a191811, 0xad28c7fe], dtype=np.uint32),
        "prime": np.array([0x36468fb3, 0x5317bade, 0x1c191811, 0x2f28c7fe], dtype=np.uint32)
    },
    {
        "normal": np.array([0xb29229c8, 0xc9024b47, 0xca77844a, 0xa4aa1647], dtype=np.uint32),
        "prime": np.array([0x329229c8, 0x4b024b47, 0x4c77844a, 0x26aa1647], dtype=np.uint32)
    },
    {
        "normal": np.array([0xc2627638, 0x59120d4f, 0x1252ff58, 0x04e627e8], dtype=np.uint32),
        "prime": np.array([0x42627638, 0xdb120d4f, 0x9452ff58, 0x86e627e8], dtype=np.uint32)
    },
    {
        "normal": np.array([0xba8c16f5, 0x09336cca, 0x0a7e15df, 0x5d6d559e], dtype=np.uint32),
        "prime": np.array([0x3a8c16f5, 0x8b336cca, 0x8c7e15df, 0xdf6d559e], dtype=np.uint32)
    },
    {
        "normal": np.array([0x895ff42d, 0xb066f615, 0xd2898682, 0xdce35664], dtype=np.uint32),
        "prime": np.array([0x095ff42d, 0x3266f615, 0x54898682, 0x5ee35664], dtype=np.uint32)
    }
]


def normal_block1_step20(m, q, ivs):
    """計算前 20 步的 MD5 狀態，涵蓋第一輪與第二輪前 4 步[cite: 3]"""
    a, b, c, d = ivs[0], ivs[1], ivs[2], ivs[3]
    T = [int(4294967296 * abs(math.sin(i))) & 0xFFFFFFFF for i in range(1, 65)]
    S = ([7, 12, 17, 22] * 4 +
         [5, 9, 14, 20] * 4 +
         [4, 11, 16, 23] * 4 +
         [6, 10, 15, 21] * 4)

    def F(x, y, z):
        return (x & y) | (~x & z)

    def G(x, y, z):
        return (x & z) | (y & ~z)

    def left_rot(x, n):
        return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF

    for i in range(20):
        if i < 16:
            f = F(b, c, d)
            g = i
        else:
            f = G(b, c, d)
            g = (5 * i + 1) % 16

        temp = (a + f + m[g] + T[i]) & 0xFFFFFFFF
        a = (b + left_rot(temp, S[i])) & 0xFFFFFFFF
        q[i] = a
        a, b, c, d = d, a, b, c


def print_horizontal_bit_matrix(q_array, title_name):
    reg_order = (['a', 'd', 'c', 'b'] * 5)[:len(q_array)]
    print("\n" + "=" * 70)
    print(f" {title_name} ({len(q_array)} rows x 32 columns)")
    print("=" * 70)
    for i in range(len(q_array)):
        bin_str = f"{q_array[i]:032b}"
        formatted_bits = " ".join(bin_str[j:j + 4] for j in range(0, 32, 4))
        reg_name = reg_order[i] if i < len(reg_order) else '?'
        print(f"q[{i:<2}]({reg_name}) | {formatted_bits}")


def main():
    try:
        idx = int(input("? "))
        if idx < 0 or idx > 4:
            idx = 0
    except ValueError:
        idx = 0

    st = time.time()
    print(f'executing with IV index {idx}......')

    ivs_normal = HARDCODED_IVS[idx]["normal"]
    ivs_prime = HARDCODED_IVS[idx]["prime"]

    print(
        f"IVs (Normal): a=0x{ivs_normal[0]:08x}, b=0x{ivs_normal[1]:08x}, c=0x{ivs_normal[2]:08x}, d=0x{ivs_normal[3]:08x}")
    print(
        f"IVs (Prime) : a=0x{ivs_prime[0]:08x}, b=0x{ivs_prime[1]:08x}, c=0x{ivs_prime[2]:08x}, d=0x{ivs_prime[3]:08x}")

    with np.errstate(over='ignore', under='ignore'):
        delta_h1_calculated = ivs_prime - ivs_normal

    print("\n" + "-" * 70)
    print("Delta H1 (ivs_prime - ivs_normal):")
    print(f"da = 0x{delta_h1_calculated[0]:08x}")
    print(f"db = 0x{delta_h1_calculated[1]:08x}")
    print(f"dc = 0x{delta_h1_calculated[2]:08x}")
    print(f"dd = 0x{delta_h1_calculated[3]:08x}")
    print("-" * 70 + "\n")


    m1_rand = np.random.randint(0, 4294967295, size=14, dtype=np.uint32)
    m1_input = m1_rand.copy()

    # q_dummy 依然保留 14 長度
    q_dummy = np.zeros(14, dtype=np.uint32)


    m1 = block1(m1_input, q_dummy, ivs_normal, np.uint32(1))


    q_normal = np.zeros(20, dtype=np.uint32)
    normal_block1_step20(m1, q_normal, ivs_normal)


    delta_m1 = np.zeros(16, dtype=np.uint32)
    delta_m1[4] = np.uint32(1) << np.uint32(31)  # +2^31
    delta_m1[11] = np.uint32(0xFFFF8000)  # -2^15
    delta_m1[14] = np.uint32(1) << np.uint32(31)  # +2^31

    m1_prime = m1 + delta_m1

    q_prime = np.zeros(20, dtype=np.uint32)
    normal_block1_step20(m1_prime, q_prime, ivs_prime)

    print("\n" + "=" * 70)
    print("Message Blocks")
    print("=" * 70)
    for i in range(16):
        diff_flag = " <--" if m1[i] != m1_prime[i] else ""
        print(f"m[{i:<2}] | m1: 0x{m1[i]:08x} | m1': 0x{m1_prime[i]:08x}{diff_flag}")

    print("\n" + "=" * 70)
    print("differential")
    print("=" * 70)
    reg_order = (['a', 'd', 'c', 'b'] * 5)[:20]
    for i in range(20):
        xor_diff = q_normal[i] ^ q_prime[i]
        with np.errstate(over='ignore', under='ignore'):
            mod_diff = np.uint32(q_prime[i] - q_normal[i])
        print(
            f"q[{i:<2}] | {reg_order[i]:<3} | 0x{q_normal[i]:08x} | 0x{q_prime[i]:08x} | 0x{xor_diff:08x} | 0x{mod_diff:08x}")

    print_horizontal_bit_matrix(q_normal, "bits of q")
    xor_array = q_normal ^ q_prime
    print_horizontal_bit_matrix(xor_array, "xor differential")
    print(f'\n Time: {time.time() - st:.2f} sec')


if __name__ == "__main__":
    main()