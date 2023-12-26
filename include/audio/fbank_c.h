#ifndef _LV_FBANK_C_H_
#define _LV_FBANK_C_H_

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct tag_fbank_t fbank_t;

int fbank_query_mem(int num_bins, int frame_length);
int fbank_query_shm(int num_bins, int frame_length);
fbank_t* fbank_init(void* mem_addr, void* shm_addr, int num_bins, int sample_rate, int frame_length, int frame_shift);
void fbank_uninit(fbank_t *svr);
int fbank_compute(fbank_t* svr, float* wave, float* feat, int num_samples);

#ifdef __cplusplus
}
#endif 

#endif // _LV_FBANK_C_H_