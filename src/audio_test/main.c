#include <stdio.h>
#include <stdlib.h>
#include "test.h"

int main(int argc, char** argv)
{
    printf("main\n");
    test_init();
    test_fft();
    test_fbank();
    test_uninit();
    return 0;
}