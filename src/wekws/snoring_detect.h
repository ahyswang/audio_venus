#ifndef _LV_SNORING_DETECT_C_H_
#define _LV_SNORING_DETECT_C_H_

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
    unsigned classes_num;
    unsigned num_bins;  //64
    unsigned sample_rate;
    unsigned max_frame_num; // 64
    unsigned frame_length; //64ms
    unsigned frame_shift; //32ms
    unsigned qvalue;
    unsigned res_size;
    void* res_addr;
    void* extra1; //for debug
    void* extra2; //for debug
} snoring_detect_conf_t;

typedef struct tag_snoring_detect_t snoring_detect_t;

unsigned snoring_detect_query_mem(snoring_detect_conf_t *conf);
unsigned snoring_detect_query_shm(snoring_detect_conf_t *conf);
snoring_detect_t* snoring_detect_init(void* mem_addr, void* shm_addr, snoring_detect_conf_t *conf);
void snoring_detect_uninit();
int snoring_detect_process(snoring_detect_t *svr, void* wave, void* classes, unsigned num_samples);

#ifdef __cplusplus
}
#endif 

#endif // _LV_SNORING_DETECT_C_H_