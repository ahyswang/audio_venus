#include <stdio.h>
#include <stdlib.h>
#include "audio/fbank_c.h"
#include "test.h"

void test_fbank()
{
    void *p_mem, *p_shm;

    int num_samples = 1600;//16000;
    int num_bins = 80;// 80 dim fbank
    int sample_rate = 16000;// 16k sample rate
    int frame_length = sample_rate / 1000 * 25;// frame length 25ms
    int frame_shift = sample_rate / 1000 * 10;// frame shift 10ms
    int num_frames = num_samples/frame_shift;
    int num_frames_2 = 0;
    fbank_t *fb;

    void *mem_addr, *shm_addr;
    unsigned mem_size, shm_size;
    float *wave, *fea;

    LOG("func:%s\n", __FUNCTION__);

    mem_size = fbank_query_mem(num_bins, frame_length);
    shm_size = fbank_query_shm(num_bins, frame_length);
    printf("mem_size:%d, shm_size:%d\n", mem_size, shm_size);

    p_mem       = g_mem_addr;
    mem_addr    = (void*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem       = (char*)p_mem + ALIGNED_SIZE(mem_size, ALIGNMENT_SIZE);
    wave        = (void*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem       = (char*)p_mem + ALIGNED_SIZE(num_samples*sizeof(float), ALIGNMENT_SIZE);
    fea         = (void*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem       = (char*)p_mem + ALIGNED_SIZE(num_bins*sizeof(float), ALIGNMENT_SIZE);
    
    p_shm       = g_shm_addr;
    shm_addr    = (void*)ALIGNED_ADDR(p_shm, ALIGNMENT_SIZE);
    p_shm       = (char*)p_shm + ALIGNED_SIZE(shm_size, ALIGNMENT_SIZE);

    ASSERT((uintptr_t)(p_mem) - (uintptr_t)(mem_addr) <= g_mem_size);
    ASSERT((uintptr_t)(p_shm) - (uintptr_t)(shm_addr) <= g_shm_size);

    fb = fbank_init(mem_addr, shm_addr, num_bins, sample_rate, frame_length, frame_shift);

#if 1    
    for (int i = 0; i < num_samples; i++)
    {
        wave[i] = i%256;
    }
#else 
    fread(wave, num_samples*sizeof(float), 1, stdin); 
    for(int i =0; i < 10; i++)
    {
        printf("%f, ", (wave+num_samples/2)[i]);
    }
    printf("\n");
#endif 

    num_frames_2 = fbank_compute(fb, wave, fea, num_samples);
    num_frames_2 = 2;
    for (int i = 0; i < num_frames_2; i++)
    {
        printf("%d:", i);
        for (int k = 0; k < num_bins; k++)
        {
            printf("%f, ", fea[i*num_bins+k]);
        }
        printf("\n");
    }
}
