import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
from block0 import (
    block0, left_rotate, phi1,
    MZ3, MO3, MZ4, MO4, MF4, MZ5, MO5, MZ6, MO6, MF6,
    MZ7, MO7, MZ8, MO8, MZ9, MO9, MF9, MZ10, MO10,
    MZ11, MO11, MF11, MZ12, MO12, MF12, MZ13, MO13,
    MZ14, MO14, MZ15, MO15, MZ16, MO16
)


MZ = [np.uint32(0)] * 16
MO = [np.uint32(0)] * 16
MF = [np.uint32(0)] * 16

MZ[2], MO[2] = MZ3, MO3
MZ[3], MO[3], MF[3] = MZ4, MO4, MF4
MZ[4], MO[4] = MZ5, MO5
MZ[5], MO[5], MF[5] = MZ6, MO6, MF6
MZ[6], MO[6] = MZ7, MO7
MZ[7], MO[7] = MZ8, MO8
MZ[8], MO[8], MF[8] = MZ9, MO9, MF9
MZ[9], MO[9] = MZ10, MO10
MZ[10], MO[10], MF[10] = MZ11, MO11, MF11
MZ[11], MO[11], MF[11] = MZ12, MO12, MF12
MZ[12], MO[12] = MZ13, MO13
MZ[13], MO[13] = MZ14, MO14
MZ[14], MO[14] = MZ15, MO15
MZ[15], MO[15] = MZ16, MO16

def normal_block0(m, q):
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

def print_horizontal_bit_matrix(q_array, title_name):
    reg_order = ['a', 'd', 'c', 'b'] * 4
    print("\n" + "=" * 70)
    print(f" {title_name} (16 rows x 32 columns)")
    print("=" * 70)
    for i in range(16):
        bin_str = f"{q_array[i]:032b}"
        formatted_bits = " ".join(bin_str[j:j+4] for j in range(0, 32, 4))
        print(f"q[{i:<2}]({reg_order[i]}) | {formatted_bits}")

def check_with_imported_masks(q_array):
    reg_order = ['a', 'd', 'c', 'b'] * 4
    all_passed = True
    
    print("\n" + "=" * 70)
    print(" " * 15 + "check if message modification valid")
    print("=" * 70)
    
    for i in range(16):
        mismatches = []
        for bit_idx in range(32):
            bit_mask = np.uint32(1) << np.uint32(bit_idx)
            bit_pos = bit_idx + 1        
            actual_bit = (q_array[i] >> np.uint32(bit_idx)) & np.uint32(1)
            if (MZ[i] & bit_mask) != 0 and actual_bit != 0:
                mismatches.append(f"B{bit_pos}(MustBe:0,Act:1)")
            if (MO[i] & bit_mask) != 0 and actual_bit != 1:
                mismatches.append(f"B{bit_pos}(MustBe:1,Act:0)")
            if i > 0 and (MF[i] & bit_mask) != 0:
                prev_bit = (q_array[i-1] >> np.uint32(bit_idx)) & np.uint32(1)
                if actual_bit != prev_bit:
                    mismatches.append(f"B{bit_pos}(MustMatchPrev,Prev:{prev_bit},Act:{actual_bit})")
        
        if mismatches:
            all_passed = False
            print(f"q[{i:<2}]({reg_order[i]}) mismatch -> {', '.join(mismatches)}")
        else:
            if MZ[i] != 0 or MO[i] != 0 or MF[i] != 0:
                print(f"q[{i:<2}]({reg_order[i]}) pass")
                
    print("-" * 70)
    
def main():
    m_rand = np.random.randint(0, 4294967295, size=16, dtype=np.uint32)
    m0 = m_rand.copy()
    q_dummy = np.zeros(16, dtype=np.uint32)
    block0(m0, q_dummy, np.uint32(1), np.uint32(0))
    delta_m0 = np.zeros(16, dtype=np.uint32)
    delta_m0[4]  = np.uint32(1) << np.uint32(31)  # 2^31
    delta_m0[11] = np.uint32(1) << np.uint32(15)  # 2^15
    delta_m0[14] = np.uint32(1) << np.uint32(31)  # 2^31
    m0_prime = m0 + delta_m0
    q = np.zeros(16, dtype=np.uint32)
    q_prime = np.zeros(16, dtype=np.uint32)
    normal_block0(m0, q)
    normal_block0(m0_prime, q_prime)
    print("=" * 70)
    print("state differential (xor & modular diff)")
    print("=" * 70)
    
    reg_order = ['a', 'd', 'c', 'b'] * 4 
    for i in range(16):
        xor_diff = q[i] ^ q_prime[i]
        with np.errstate(over='ignore', under='ignore'):
            mod_diff = np.uint32(q_prime[i] - q[i])
        print(f"q[{i:<2}] | {reg_order[i]:<3} | 0x{q[i]:08x} | 0x{q_prime[i]:08x} | 0x{xor_diff:08x} | 0x{mod_diff:08x}")
    print_horizontal_bit_matrix(q, "bits of q (from M0)")
    xor_array = q ^ q_prime
    print_horizontal_bit_matrix(xor_array, "xor differential matrix (q ^ q')")
    check_with_imported_masks(q)

if __name__ == "__main__":
    main()
