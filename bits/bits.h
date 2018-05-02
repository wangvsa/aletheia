#ifndef BITS_H
#define BITS_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void doubleToBinary(double d, char *p) {
    union { double d; uint64_t i; } u;
    u.d = d;
    int i;
    for (i = 0; i < 64; i++) {
        if (u.i % 2)  p[63-i] = '1';
        else p[63-i] = '0';
        u.i >>= 1;
    }
}

double binaryToDouble(const char* p) {
    unsigned long long x = 0;
    for (; *p; ++p) {
        x = (x << 1) + (*p - '0');
    }
    double d;
    memcpy(&d, &x, 8);
    return d;
}

double bit_flip(double d, int pos) {
    char p[65] = {0};
    doubleToBinary(d, p);
    printf("%s\n", p);
    if(p[pos] == '1')
        p[pos] = '0';
    else
        p[pos] = '1';
    double error = binaryToDouble(p);
    return error;
}

#endif
