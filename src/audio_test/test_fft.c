#include <stdio.h>
#include <string.h>

#include "audio/fft.h"
#include "test.h"

extern unsigned g_mem_size;
extern unsigned g_shm_size;
extern void* g_mem_addr;
extern void* g_shm_addr;

#define FFT_SIZE (16)
/*
x = [1:16]
yk = fft([1:16])
*/
static const float x_ref[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
static const float yk_ref[] = {
    136,0, -8,40.218715937006785, -8,19.313708498984759, -8,11.972846101323912, -8,8, -8,5.3454291033543893, -8,3.3137084989847612, -8,1.5912989390372658, -8,0, -8,-1.5912989390372658, -8,-3.3137084989847612, -8,-5.3454291033543893, -8,-8, -8,-11.972846101323912, -8,-19.313708498984759, -8,-40.218715937006785,
};

void test_fft()
{   
    int fft_size = FFT_SIZE;
    float sintbl[FFT_SIZE+FFT_SIZE/4];
    int revtbl[FFT_SIZE];
    float data[FFT_SIZE];
    float fft_real[FFT_SIZE];
    float fft_img[FFT_SIZE];
    float yk[FFT_SIZE*2]; 

    LOG("func:%s\n", __FUNCTION__);

    make_sintbl(fft_size, sintbl);
    make_bitrev(fft_size, revtbl);

    for (int i = 0; i < fft_size; i++)
    {
        data[i] = i+1;
    }
    memset(fft_img, 0, sizeof(float) * fft_size);
    memset(fft_real, 0, sizeof(float) * fft_size);
    memcpy(fft_real, data, sizeof(float) * fft_size);
    fft(revtbl, sintbl, fft_real, fft_img, fft_size);
    for (int i = 0; i < fft_size; i++)
    {
        yk[i*2 + 0] = fft_real[i];
        yk[i*2 + 1] = fft_img[i];
    }
    for (int i = 0; i < fft_size; i++)
    {
        LOGD("%f, %f, ", fft_real[i], fft_img[i]);
    }
    float diff_mse = 0;
    float diff_max = 0;
    for (int i = 0; i < fft_size*2; i++)
    {
        float a = fabs(yk[i] - yk_ref[i]);
        diff_mse += a;
        if (a > diff_max) {
            diff_max = a;
        }
    }
    diff_mse = diff_mse / (fft_size*2); 
    LOGD("diff_mse:%f, diff_max:%f\n", diff_mse, diff_max);
}