import time
import warnings
import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)

from block0 import block0, left_rotate, phi1, phi2, phi3, phi4

IV_A = np.uint32(0x67452301)
IV_B = np.uint32(0xefcdab89)
IV_C = np.uint32(0x98badcfe)
IV_D = np.uint32(0x10325476)

K_TABLE = np.array([
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
], dtype=np.uint32)

S_TABLE = np.array([
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
], dtype=np.uint32)

M_IDX = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12,
    5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2,
    0, 7, 14, 5, 12, 3, 10, 1, 8, 15, 6, 13, 4, 11, 2, 9
], dtype=np.uint32)

def compute_md5_64_steps(m):
    a, b, c, d = IV_A, IV_B, IV_C, IV_D
    q = np.zeros(64, dtype=np.uint32)

    for i in range(64):
        if i < 16:
            phi = phi1
        elif i < 32:
            phi = phi2
        elif i < 48:
            phi = phi3
        else:
            phi = phi4

        if i % 4 == 0:
            a = b + left_rotate((a + phi(b, c, d) + m[M_IDX[i]] + K_TABLE[i]), S_TABLE[i])
            q[i] = a
        elif i % 4 == 1:
            d = a + left_rotate((d + phi(a, b, c) + m[M_IDX[i]] + K_TABLE[i]), S_TABLE[i])
            q[i] = d
        elif i % 4 == 2:
            c = d + left_rotate((c + phi(d, a, b) + m[M_IDX[i]] + K_TABLE[i]), S_TABLE[i])
            q[i] = c
        elif i % 4 == 3:
            b = c + left_rotate((b + phi(c, d, a) + m[M_IDX[i]] + K_TABLE[i]), S_TABLE[i])
            q[i] = b

    return q

def print_horizontal_bit_matrix(q_array, title_name, labels=None):
    if labels is None:
        reg_order = ['a', 'd', 'c', 'b'] * 16
    else:
        reg_order = labels

    print("\n" + "=" * 70)
    print(f" {title_name} ({len(q_array)} rows x 32 columns)")
    print("=" * 70)
    for i in range(len(q_array)):
        bin_str = f"{q_array[i]:032b}"
        formatted_bits = " ".join(bin_str[j:j+4] for j in range(0, 32, 4))
        prefix = f"q[{i:<2}]({reg_order[i]})" if labels is None else f"{reg_order[i]:<7}"
        print(f"{prefix} | {formatted_bits}")

def main():
    m_rand = np.random.randint(0, 4294967295, size=16, dtype=np.uint32)
    q_dummy = np.zeros(16, dtype=np.uint32)
    print("Executing...")   
    start_time = time.time()
    m0 = block0(m_rand.copy(), q_dummy, np.uint32(0))
    elapsed = time.time() - start_time    
    print(f"Found! Time: {elapsed:.2f} seconds")
    q_normal = compute_md5_64_steps(m0)
    delta_m0 = np.zeros(16, dtype=np.uint32)
    delta_m0[4]  = np.uint32(1) << np.uint32(31)  # 2^31
    delta_m0[11] = np.uint32(1) << np.uint32(15)  # 2^15
    delta_m0[14] = np.uint32(1) << np.uint32(31)  # 2^31  
    m0_prime = m0 + delta_m0
    q_prime = compute_md5_64_steps(m0_prime)   
    out_labels = ['aa', 'bb', 'cc', 'dd']
    final_normal = np.array([q_normal[60] + IV_A, q_normal[63] + IV_B, q_normal[62] + IV_C, q_normal[61] + IV_D], dtype=np.uint32)
    final_prime  = np.array([q_prime[60] + IV_A, q_prime[63] + IV_B, q_prime[62] + IV_C, q_prime[61] + IV_D], dtype=np.uint32)

    for i in range(16):
        diff_flag = " <-- " if m0[i] != m0_prime[i] else ""
        print(f"m[{i:<2}] | m0: 0x{m0[i]:08x} | m0': 0x{m0_prime[i]:08x}{diff_flag}")
    
    reg_order = ['a', 'd', 'c', 'b'] * 16 
    print('\n')
    print(f"{'Step':<6} | {'Reg':<3} | {'q_normal':<10} | {'q_prime':<10} | {'XOR Diff':<10} | {'Mod Diff':<10}")
    print("-" * 70)
    for i in range(64):
        xor_diff = q_normal[i] ^ q_prime[i]
        with np.errstate(over='ignore', under='ignore'):
            mod_diff = np.uint32(q_prime[i] - q_normal[i])
        highlight = " *" if xor_diff != 0 else ""
        print(f"q[{i:<2}] {highlight:<2}| {reg_order[i]:<3} | 0x{q_normal[i]:08x} | 0x{q_prime[i]:08x} | 0x{xor_diff:08x} | 0x{mod_diff:08x}")
    print('\n')
    print(f"{'Reg':<6} | {'Normal':<10} | {'Prime':<10} | {'XOR Diff':<10} | {'Mod Diff':<10}")
    print("-" * 70)
    for i in range(4):
        xor_diff = final_normal[i] ^ final_prime[i]
        with np.errstate(over='ignore', under='ignore'):
            mod_diff = np.uint32(final_prime[i] - final_normal[i])
        highlight = " *" if xor_diff != 0 else ""
        print(f"{out_labels[i]:<4} {highlight:<1} | 0x{final_normal[i]:08x} | 0x{final_prime[i]:08x} | 0x{xor_diff:08x} | 0x{mod_diff:08x}")
        
    print_horizontal_bit_matrix(q_normal, "Original M0 State Matrix (Bits of q from M0)")
    print_horizontal_bit_matrix(q_normal ^ q_prime, "XOR Differential Matrix (q ^ q')")
    
    print_horizontal_bit_matrix(final_normal, "Final Output Registers Matrix (aa, bb, cc, dd)", labels=out_labels)
    print_horizontal_bit_matrix(final_normal ^ final_prime, "Final Output XOR Differential Matrix (aa^aa', bb^bb', ...)", labels=out_labels)

if __name__ == "__main__":
    main()
