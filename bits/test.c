#include "bits.h"
#include <math.h>

int main() {
    //char *p = doubleToBinary(0.999);
    //printf("%s\n", p);
    //printf("%f\n", binaryToDouble(p));
    double error = bit_flip(1.0, 1);
    printf("%e\n", error);
    printf("%d\n", isnormal(error));
    return 0;
}
