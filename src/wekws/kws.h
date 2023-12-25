#ifndef _LV_KWS_C_H_
#define _LV_KWS_C_H_

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
    int keyword_num;
    int num_bins;
    int sample_rate;
    int frame_length;
    int frame_shift;
    int qvalue;
    void* res_addr;
    int res_size;
    void* mean;
    void* istd;
    void* extra;
} kws_conf_t;

typedef struct tag_kws_t kws_t;

int kws_query_mem(kws_conf_t *conf);
int kws_query_shm(kws_conf_t *conf);
kws_t* kws_init(void* mem_addr, void* shm_addr, kws_conf_t *conf);
void kws_uninit();
int kws_process(kws_t *svr, float* wave, float* classes, int num_samples);

#ifdef __cplusplus
}
#endif 

#endif // _LV_KWS_C_H_