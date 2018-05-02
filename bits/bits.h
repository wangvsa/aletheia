#ifndef BITS_H
#define BITS_H

#include <cstring>
#include <stdio.h>
#include <iostream>
#include <bitset>
using namespace std;

void doubleToBinary(double d, string &str) {
    union { double d; uint64_t i; } u;
    u.d = d;
    str.clear();
    for (int i = 0; i < 64; i++) {
        if (u.i % 2)  str.push_back('1');
        else str.push_back('0');
        u.i >>= 1;
    }
    // Reverse the string since now it's backwards
    string temp(str.rbegin(), str.rend());
    str = temp;
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

double binaryToDouble(string &str) {
    return binaryToDouble(str.c_str());
}

double bit_flip(double d, int pos) {
    string str;
    doubleToBinary(d, str);
    if(str[pos] == '1')
        str[pos] = '0';
    else
        str[pos] = '1';
    double error = binaryToDouble(str);
    return error;
}

#endif
