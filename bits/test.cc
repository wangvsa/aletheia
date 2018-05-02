#include <iostream>
using namespace std;
#include "bits.h"

int main() {
    string str;
    doubleToBinary(0.999, str);
    cout<<str<<endl;
    cout<<binaryToDouble(str)<<endl;
    cout<<bit_flip(0.999, 2)<<endl;
    return 0;
}
