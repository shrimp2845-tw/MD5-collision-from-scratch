import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
from block0 import block0, left_rotate, phi1, phi2

def normal_block0_step20(m, q):
    a, b, c, d = np.uint32(0x67452301), np.uint32(0xefcdab89), np.uint32(0x98badcfe), np.uint32(0x10325476)  
    a = b + left_rotate((a + phi1(b, c, d) + m[0] + np.uint32(0xd76aa478)), np.uint32(7)); q[0] = a
    d = a + left_rotate((d + phi1(a, b, c) + m[1] + np.uint32(0xe8c7b756)), np.uint32(12)); q[1] = d
    c = d + left_rotate((c + phi1(d, a, b) + m[2] + np.uint32(0x242070db)), np.uint32(17)); q[2] = c
    b = c + left_rotate((b + phi1(c, d, a) + m[3] + np.uint32(0xc1bdceee)), np.uint32(22)); q[3] = b   
    a = b + left_rotate((a + phi1(b, c, d) + m[4] + np.uint32(0xf57c0faf)), np.uint32(7)); q[4] = a
    d = a + left_rotate((d + phi1(a, b, c) + m[5] + np.uint32(0x4787c62a)), np.uint32(12)); q[5] = d
    c = d + left_rotate((c + phi1(d, a, b) + m[6] + np.uint32(0xa8304613)), np.uint32(17)); q[6] = c
    b = c + left_rotate((b + phi1(c, d, a) + m[7] + np.uint32(0xfd469501)), np.uint32(22)); q[7] = b
    a = b + left_rotate((a + phi1(b, c, d) + m[8] + np.uint32(0x698098d8)), np.uint32(7)); q[8] = a
    d = a + left_rotate((d + phi1(a, b, c) + m[9] + np.uint32(0x8b44f7af)), np.uint32(12)); q[9] = d
    c = d + left_rotate((c + phi1(d, a, b) + m[10] + np.uint32(0xffff5bb1)), np.uint32(17)); q[10] = c
    b = c + left_rotate((b + phi1(c, d, a) + m[11] + np.uint32(0x895cd7be)), np.uint32(22)); q[11] = b
    a = b + left_rotate((a + phi1(b, c, d) + m[12] + np.uint32(0x6b901122)), np.uint32(7)); q[12] = a
    d = a + left_rotate((d + phi1(a, b, c) + m[13] + np.uint32(0xfd987193)), np.uint32(12)); q[13] = d
    c = d + left_rotate((c + phi1(d, a, b) + m[14] + np.uint32(0xa679438e)), np.uint32(17)); q[14] = c
    b = c + left_rotate((b + phi1(c, d, a) + m[15] + np.uint32(0x49b40821)), np.uint32(22)); q[15] = b
    a = b + left_rotate((a + phi2(b, c, d) + m[1] + np.uint32(0xf61e2562)), np.uint32(5)); q[16] = a
    d = a + left_rotate((d + phi2(a, b, c) + m[6] + np.uint32(0xc040b340)), np.uint32(9)); q[17] = d
    c = d + left_rotate((c + phi2(d, a, b) + m[11] + np.uint32(0x265e5a51)), np.uint32(14)); q[18] = c
    b = c + left_rotate((b + phi2(c, d, a) + m[0] + np.uint32(0xe9b6c7aa)), np.uint32(20)); q[19] = b

def print_horizontal_bit_matrix(q_array, title_name):
    reg_order = ['a', 'd', 'c', 'b'] * 5
    print("\n" + "=" * 70)
    print(f" {title_name} ({len(q_array)} rows x 32 columns)")
    print("=" * 70)
    for i in range(len(q_array)):
        bin_str = f"{q_array[i]:032b}"
        formatted_bits = " ".join(bin_str[j:j+4] for j in range(0, 32, 4))
        print(f"q[{i:<2}]({reg_order[i]}) | {formatted_bits}")

def main():
    m_rand = np.random.randint(0, 4294967295, size=16, dtype=np.uint32)
    m0 = m_rand.copy()
    q_dummy = np.zeros(16, dtype=np.uint32)
    q_returned = block0(m0, q_dummy, np.uint32(1))
    q_normal = np.zeros(20, dtype=np.uint32)
    normal_block0_step20(m0, q_normal)
    is_match = np.array_equal(q_returned, q_normal)
    print("=" * 70)
    print(f" Check q array match (block0 returned vs Normal compute): {'PASS' if is_match else 'FAIL'}")
    print("=" * 70)
    if not is_match:
        print("Warning: q_returned from block0 does not match locally calculated q_normal!")
        for i in range(20):
            if q_returned[i] != q_normal[i]:
                print(f"Mismatch at step {i}: returned=0x{q_returned[i]:08x}, normal=0x{q_normal[i]:08x}")

    delta_m0 = np.zeros(16, dtype=np.uint32)
    delta_m0[4]  = np.uint32(1) << np.uint32(31)  # 2^31
    delta_m0[11] = np.uint32(1) << np.uint32(15)  # 2^15
    delta_m0[14] = np.uint32(1) << np.uint32(31)  # 2^31
    m0_prime = m0 + delta_m0
    
    q_prime = np.zeros(20, dtype=np.uint32)
    normal_block0_step20(m0_prime, q_prime)
    
    print("\n" + "=" * 70)
    print(" Message Blocks (m0 vs m0_prime)")
    print("=" * 70)
    for i in range(16):
        diff_flag = " <--" if m0[i] != m0_prime[i] else ""
        print(f"m[{i:<2}] | m0: 0x{m0[i]:08x} | m0': 0x{m0_prime[i]:08x}{diff_flag}")

    print("\n" + "=" * 70)
    print(" state differential (xor & modular diff)")
    print("=" * 70)
    reg_order = ['a', 'd', 'c', 'b'] * 5 
    for i in range(20):
        xor_diff = q_normal[i] ^ q_prime[i]
        with np.errstate(over='ignore', under='ignore'):
            mod_diff = np.uint32(q_prime[i] - q_normal[i])
        print(f"q[{i:<2}] | {reg_order[i]:<3} | 0x{q_normal[i]:08x} | 0x{q_prime[i]:08x} | 0x{xor_diff:08x} | 0x{mod_diff:08x}")
        
    print_horizontal_bit_matrix(q_normal, "bits of q (from M0)")
    xor_array = q_normal ^ q_prime
    print_horizontal_bit_matrix(xor_array, "xor differential matrix (q ^ q')")

if __name__ == "__main__":
    main()
