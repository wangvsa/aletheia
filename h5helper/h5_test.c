#include "h5_writer.h"

int main() {
    float data[5*10];
    for( int i = 0; i < 5; i++ )
        for(int j = 0; j < 10; j++)
            data[i*10+j] = i * 20.0 + j;

    write_h5_data("test.h5", data, 5, 10);
    return 0;
}

