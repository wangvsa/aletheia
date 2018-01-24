import bitstring


# Flip a bit of a given position
# val: input, python float(64 bits), equals double in C
def bit_flip(val, pos):

    bin_str = bitstring.BitArray(float=val, length=64).bin

    l = list(bin_str)
    l[pos] = '1' if l[pos] == '0' else '0'
    flipped_str = "".join(l)

    return bitstring.BitArray(bin=flipped_str).float


# 64 bits binary string to float
# method 2, use struct pack and unpack
def as_float64(bin_str):
    bits = [int(x) for x in bin_str[::-1]]
    x = 0
    for i in range(len(bits)):
        x += bits[i]*2**i
    from struct import pack,unpack
    return unpack("d", pack("L", x))

val = 1.0
for i in range(64):
    print i, ':', bit_flip(val, i)
