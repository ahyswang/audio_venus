#include <stdio.h>
#include <stdlib.h>
#include "kws.h"
#include "test.h"

void test_kws()
{
    void *p_mem, *p_shm;
    kws_t *kws;
    kws_conf_t conf;
    
    int num_frames = 0;
    int num_bins = 40;// 80 dim fbank
    int sample_rate = 16000;// 16k sample rate
    int frame_length = sample_rate / 1000 * 25;// frame length 25ms
    int frame_shift = sample_rate / 1000 * 10;// frame shift 10ms
    int keyword_num = 2;
    
    void *mem_addr, *shm_addr;
    unsigned mem_size, shm_size;
    float *wave, *fea;

    void *res_addr, *input_addr, *meanstd_addr;
    unsigned res_size, input_size, meanstd_size;

    const char* res_path = "./bin/pack.bin";
    const char* wav_path = "./bin/input.wav";
    const char* meanstd_path = "./bin/meanstd.bin";

    load_bin(res_path, &res_addr, &res_size);
    load_bin(wav_path, &input_addr, &input_size);
    load_bin(meanstd_path, &meanstd_addr, &meanstd_size);

    LOG("func:%s\n", __FUNCTION__);

    conf.num_bins = num_bins;
    conf.sample_rate = sample_rate;
    conf.frame_length = frame_length;
    conf.frame_shift = frame_shift;
    conf.keyword_num = keyword_num;
    conf.qvalue = 4; // from onnx quant scale
    conf.res_addr = res_addr; 
    conf.res_size = res_size;
    conf.mean = (float*)meanstd_addr;
    conf.istd = (float*)meanstd_addr + num_bins;
    conf.extra = 0;

    mem_size = kws_query_mem(&conf);
    shm_size = kws_query_shm(&conf);
    printf("mem_size:%d, shm_size:%d\n", mem_size, shm_size);

    p_mem       = g_mem_addr;
    mem_addr    = (void*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem       = (char*)p_mem + ALIGNED_SIZE(mem_size, ALIGNMENT_SIZE);
    wave        = (void*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem       = (char*)p_mem + ALIGNED_SIZE(frame_shift*sizeof(float), ALIGNMENT_SIZE);
    fea         = (void*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem       = (char*)p_mem + ALIGNED_SIZE(keyword_num*sizeof(float), ALIGNMENT_SIZE);
    
    p_shm       = g_shm_addr;
    shm_addr    = (void*)ALIGNED_ADDR(p_shm, ALIGNMENT_SIZE);
    p_shm       = (char*)p_shm + ALIGNED_SIZE(shm_size, ALIGNMENT_SIZE);

    ASSERT((uintptr_t)(p_mem) - (uintptr_t)(mem_addr) <= g_mem_size);
    ASSERT((uintptr_t)(p_shm) - (uintptr_t)(shm_addr) <= g_shm_size);

    kws = kws_init(p_mem, p_shm, &conf);
    ASSERT(kws != 0);

    num_frames  = input_size / (frame_shift*sizeof(short)); 
    for (int i = 0; i < num_frames; i++){
        printf("process[%d/%d]:\n", i, num_frames);
        for (int j = 0; j < frame_shift; j++) {
            wave[j] = (float)(((short*)input_addr)[i*frame_shift+j]);
        }
        kws_process(kws, wave, fea, frame_shift);
        for (int j = 0; j < keyword_num; j++) {
            printf("keyword:%d, pred:%d\n", j, fea[j]);
        }
    }

    kws_uninit(kws);
}