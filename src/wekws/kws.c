#include "kws.h"
#include "define.h"
#include "fbank_c.h"
#include "thinker.h"

typedef struct tag_kws_t
{
    float* p_mean;
    float* p_istd;
    float* p_history;
    float* p_fea;
    float* p_classes;
    void*  p_fbank_mem;
    void*  p_fbank_shm;
    int    fbank_memsize;
    int    fbank_shmsize;
    void*  p_thinker_mem;
    void*  p_thinker_shm;
    int    thinker_memsize;
    int    thinker_shmsize;
    kws_conf_t* p_conf;
    fbank_t* p_fbank;
    thinker_t* p_thinker;
}kws_t;

static void vec_meanstdf(void* y, void* x, void* mean, void* istd, int N)
{
    float* p_x = (float *)x;
    float* p_y = (float *)y;
    float* p_mean = (float *)mean;
    float* p_istd = (float *)istd;
    for (int i = 0; i < N; i++){
        p_y[i] = (p_x[i] - p_mean[i]) * p_istd[i];
    }
}

static void vec_sigmoidf(void* y, void* x, int N)
{
    float* p_x = (float *)x;
    float* p_y = (float *)y;
    for (int i = 0; i < N; i++){
        p_y[i] = 1.0 / (1.0 + expf(-p_x[i]));
    }
}

static void vec_float2int8(void* y, void* x, int N, int qvalue)
{
    float* p_x = (float *)x;
    float* p_y = (float *)y;
    float  scale = (1<<qvalue);
    for (int i = 0; i < N; i++){
        p_y[i] = floor(p_x[i]*scale + 0.5f);
    }
}

static void vec_int82float(void* y, void* x, int N, int qvalue)
{
    float* p_x = (float *)x;
    float* p_y = (float *)y;
    float  iscale = 1.0/(1<<qvalue);
    for (int i = 0; i < N; i++){
        p_y[i] = p_x[i]*iscale; 
    }
}

static void thinker_process(kws_t* svr, kws_conf_t* p_conf)
{
    tensor_t input_tensor, output_tensor;

    input_tensor.dtype = 0;
    input_tensor.scale = p_conf->qvalue;
    input_tensor.data = svr->p_fea;
    input_tensor.dim = 3;
    input_tensor.shape[0] = 1;
    input_tensor.shape[1] = 1;
    input_tensor.shape[2] = p_conf->num_bins;
    vec_float2int8(svr->p_fea, svr->p_fea, p_conf->num_bins, input_tensor.scale);
    thinker_set_input(svr->p_thinker, 0, &input_tensor);  //quant

    thinker_forward(svr->p_thinker);

    output_tensor.data = svr->p_fea;
    output_tensor.size = p_conf->keyword_num;
    thinker_get_output(svr->p_thinker, 0, &output_tensor); //dequant
    vec_float2int8(svr->p_classes, svr->p_fea, p_conf->keyword_num, output_tensor.scale);
}

int kws_query_mem(kws_conf_t *conf)
{
    
    return 0;
}

int kws_query_shm(kws_conf_t *conf)
{
    return 0;
}

kws_t* kws_init(void* mem_addr, void* shm_addr, kws_conf_t *conf)
{
    kws_t *svr;
    void *p_mem, *p_shm;

    ASSERT(mem_addr != 0 && shm_addr != 0 && conf != 0);
    ASSERT(16000 == conf->sample_rate);
    ASSERT(25*16 == conf->frame_length);
    ASSERT(10*16 == conf->frame_shift);
    ASSERT(40 == conf->num_bins);
    ASSERT(4 == conf->keyword_num);

    p_mem               = mem_addr;
    p_shm               = shm_addr; 
    svr                 = (kws_t*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE); 
    p_mem               = (char*)p_mem + ALIGNED_SIZE(sizeof(kws_t), ALIGNMENT_SIZE);

    svr->fbank_memsize  = fbank_query_mem(conf->num_bins, conf->frame_length);
    svr->fbank_shmsize  = fbank_query_shm(conf->num_bins, conf->frame_length);
    svr->p_mean         = conf->mean;
    svr->p_istd         = conf->istd;
    svr->thinker_memsize = thinker_query_mem(conf->res_addr, conf->res_size);
    svr->thinker_shmsize = thinker_query_shm(conf->res_addr, conf->res_size);
   
    svr->p_history      = (float*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem               = (char*)p_mem + ALIGNED_SIZE(conf->frame_length, ALIGNMENT_SIZE);
    svr->p_fea          = (float*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem               = (char*)p_mem + ALIGNED_SIZE(conf->num_bins, ALIGNMENT_SIZE);
    svr->p_classes      = (float*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem               = (char*)p_mem + ALIGNED_SIZE(conf->keyword_num, ALIGNMENT_SIZE);
    svr->p_fbank_mem    = (void*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem               = (char*)p_mem + ALIGNED_SIZE(svr->fbank_memsize, ALIGNMENT_SIZE);
    svr->p_thinker_mem  = (void*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem               = (char*)p_mem + ALIGNED_SIZE(svr->thinker_memsize, ALIGNMENT_SIZE);

    svr->p_fbank_shm    = (void*)ALIGNED_ADDR(p_shm, ALIGNMENT_SIZE);
    //p_shm               = (char*)p_mem + ALIGNED_SIZE(svr->fbank_shmsize, ALIGNMENT_SIZE);
    svr->p_thinker_shm  = (void*)ALIGNED_ADDR(p_shm, ALIGNMENT_SIZE);
    p_shm               = (char*)p_shm + ALIGNED_SIZE(svr->thinker_shmsize, ALIGNMENT_SIZE);

    ASSERT((uintptr_t)(p_mem) - (uintptr_t)(mem_addr) <= kws_query_mem(conf));
    ASSERT((uintptr_t)(p_shm) - (uintptr_t)(shm_addr) <= kws_query_shm(conf));

    svr->p_fbank        = fbank_init(svr->p_fbank_mem, svr->p_fbank_shm, conf->num_bins, conf->sample_rate, conf->frame_length, conf->frame_shift);
    svr->p_thinker      = thinker_init(svr->p_thinker_mem, svr->p_thinker_shm, conf->res_addr, conf->res_size);

    return svr;
}

void kws_uninit()
{

}

int kws_process(kws_t *svr, float* wave, float* classes, int num_samples)
{
    kws_conf_t* p_conf = svr->p_conf;
    int ret = 0;
    int num_frames, num_frames_2;
    
    if (num_samples <= 0 || (num_samples % p_conf->frame_shift != 0)){
        printf("num_samples error, num_samples:%d\n", num_samples);
        return -1;
    }

    num_frames = num_samples / p_conf->frame_shift;
    
    for (int k = 0; k < num_frames; k++){
        memmove(svr->p_history, svr->p_history + p_conf->frame_shift, 
            (p_conf->frame_length - p_conf->frame_shift)*sizeof(float));
        memcpy(svr->p_history + (p_conf->frame_length - p_conf->frame_shift), 
            wave + k*p_conf->frame_shift, p_conf->frame_shift*sizeof(float));
        // compute fbank
        num_frames_2 = fbank_compute(svr->p_fbank, svr->p_history, svr->p_fea, p_conf->frame_length);
        // mean/istd
        vec_meanstdf(svr->p_fea, svr->p_fea, svr->p_mean, svr->p_istd, p_conf->num_bins);
        // backbone forward
        thinker_process(svr, p_conf);
        // classifier
        vec_sigmoidf(svr->p_classes, svr->p_classes, p_conf->keyword_num);
    }
    
    return 0;
}