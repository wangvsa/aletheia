import bitstring


# Flip a bit of a given position
# bin_str: input, string
def bit_flip(bin_str, pos):
    l = list(bin_str)
    l[pos] = '1' if l[pos] == '0' else '0'
    flip_str = "".join(l)
    return flip_str

# 64 bits binary string to float
# method 1, use bitstring
def bin_to_float(bin_str):
    f = bitstring.BitArray(bin=bin_str)
    return f.float

# 64 bits binary string to float
# method 2, use struct pack and unpack
def as_float64(bin_str):
    bits = [int(x) for x in bin_str[::-1]]
    x = 0
    for i in range(len(bits)):
        x += bits[i]*2**i
    from struct import pack,unpack
    return unpack("d", pack("L", x))


f = bitstring.BitArray(float=1.0, length=64)
bin_str = f.bin
print "before:", bin_str, ", float:", bin_to_float(bin_str)

for i in range(64):
    flip_str = bit_flip(bin_str, i)
    print i, ',', bin_to_float(flip_str), ',', as_float64(flip_str)
