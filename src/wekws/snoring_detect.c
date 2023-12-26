#include "snoring_detect.h"
#include "define.h"
#include "fbank_c.h"
#include "thinker.h"

#define MAGIC 0x20231225

typedef struct tag_snoring_detect_t
{
    unsigned magic;
    unsigned fbank_memsize;
    unsigned fbank_shmsize;
    unsigned thinker_memsize;
    unsigned thinker_shmsize;

    float* p_mean;
    float* p_istd;
    float* p_history;
    float* p_fea;
    float* p_classes;
    void*  p_fbank_mem;
    void*  p_fbank_shm;
    void*  p_thinker_mem;
    void*  p_thinker_shm;
    
    snoring_detect_conf_t* p_conf;
    fbank_t* p_fbank;
    thinker_t* p_thinker;
}snoring_detect_t;

static void vec_meanstdf(void* y, void* x, void* mean, void* istd, unsigned N)
{
    float* p_x = (float *)x;
    float* p_y = (float *)y;
    float* p_mean = (float *)mean;
    float* p_istd = (float *)istd;
    unsigned i;
    for (i = 0; i < N; i++){
        p_y[i] = (p_x[i] - p_mean[i]) * p_istd[i];
    }
}

static void vec_norm(void* y, void* x, unsigned N)
{
    float* p_x = (float *)x;
    float* p_y = (float *)y;
    float mean = 0.0f;
    float istd = 0.0f;;
    unsigned i;
    for (i = 0; i < N; i++) {
        mean += p_x[i];
    }
    mean = mean / N;
    for (i = 0; i < N; i++) {
        istd += (p_x[i] - mean) * (p_x[i] - mean);
    }
    istd = istd / N;
    istd = sqrtf(istd);
    istd = 1.0f/(istd + 1.0e-6f);
    for (i = 0; i < N; i++) {
        p_y[i] = (p_x[i] - mean) * istd;
    }
}

static void vec_sigmoidf(void* y, void* x, unsigned N)
{
    float* p_x = (float *)x;
    float* p_y = (float *)y;
    unsigned i;
    for (i = 0; i < N; i++){
        p_y[i] = 1.0 / (1.0 + expf(-p_x[i]));
    }
}

static void vec_float2int8(void* y, void* x, int N, unsigned qvalue)
{
    float* p_x = (float *)x;
    char* p_y = (char *)y;
    float  scale = (1<<qvalue);
    unsigned i;
    for (i = 0; i < N; i++){
        p_y[i] = floor(p_x[i]*scale + 0.5f);
        p_y[i] = p_y[i]<-128?128:p_y[i];
        p_y[i] = p_y[i]>127?127:p_y[i];
    }
}

static void vec_int82float(void* y, void* x, int N, unsigned qvalue)
{
    char* p_x = (char *)x;
    float* p_y = (float *)y;
    float  iscale = 1.0/(1<<qvalue);
    unsigned i;
    for (i = 0; i < N; i++){
        p_y[i] = p_x[i]*iscale; 
    }
}

static void vec_softmaxf(void* y, void* x, unsigned N)
{
    float* p_x = (float *)x;
    float* p_y = (float *)y;
    float m = 0.0f;
    float sum = 0.0f;
    unsigned i;
    for (i = 0; i < N; i++){
        if (m < p_x[i]) {
            m = p_x[i];
        }
    }
    for (i = 0; i < N; i++) {
        p_y[i] = expf(p_x[i] - m);
        sum += p_y[i];
    }
    for (i = 0; i < N; i++) {
        p_y[i] = p_y[i] / sum;
    }
}

static void thinker_process(snoring_detect_t* svr, snoring_detect_conf_t* p_conf, unsigned num_frames)
{
    tensor_t input_tensor, output_tensor;

    input_tensor.dtype = 0;
    input_tensor.scale = p_conf->qvalue;
    input_tensor.data = svr->p_fea;
    input_tensor.dim = 4;
    input_tensor.shape[0] = 1;
    input_tensor.shape[1] = 1;
    input_tensor.shape[2] = num_frames;
    input_tensor.shape[3] = p_conf->num_bins;
    vec_float2int8(svr->p_fea, svr->p_fea, num_frames*p_conf->num_bins, input_tensor.scale);
    thinker_set_input(svr->p_thinker, 0, &input_tensor);  //quant

    thinker_forward(svr->p_thinker);

    output_tensor.data = svr->p_fea; // reuse
    output_tensor.size = p_conf->classes_num;
    thinker_get_output(svr->p_thinker, 0, &output_tensor); //dequant
    vec_int82float(svr->p_classes, svr->p_fea, p_conf->classes_num, output_tensor.scale);
}

unsigned snoring_detect_query_mem(snoring_detect_conf_t *conf)
{
    unsigned ret = 0;
    ASSERT(conf);
    ret = ALIGNMENT_SIZE;
    ret += ALIGNED_SIZE(sizeof(snoring_detect_t), ALIGNMENT_SIZE);
    ret += ALIGNED_SIZE(conf->max_frame_num*conf->num_bins*sizeof(float), ALIGNMENT_SIZE);
    ret += ALIGNED_SIZE(conf->classes_num*sizeof(float), ALIGNMENT_SIZE);
    ret += ALIGNED_SIZE(fbank_query_mem(conf->num_bins, conf->frame_length), ALIGNMENT_SIZE);
    ret += ALIGNED_SIZE(thinker_query_mem(conf->res_addr, conf->res_size), ALIGNMENT_SIZE);
    return ret;
}

unsigned snoring_detect_query_shm(snoring_detect_conf_t *conf)
{
    unsigned ret = 0;
    ASSERT(conf);
    ret = ALIGNMENT_SIZE;
    ret = ALIGNED_SIZE(fbank_query_shm(conf->num_bins, conf->frame_length), ALIGNMENT_SIZE);
    ret = ALIGNED_SIZE(thinker_query_shm(conf->res_addr, conf->res_size), ALIGNMENT_SIZE);
    return ret;
}

snoring_detect_t* snoring_detect_init(void* mem_addr, void* shm_addr, snoring_detect_conf_t *conf)
{
    snoring_detect_t *svr;
    void *p_mem, *p_shm;
    ASSERT(mem_addr != 0 && shm_addr != 0 && conf != 0);
    ASSERT(16000 == conf->sample_rate);
    ASSERT(64*16 == conf->frame_length);
    ASSERT(32*16 == conf->frame_shift);
    ASSERT(64 == conf->num_bins);
    ASSERT(2 == conf->classes_num);
    ASSERT(64 == conf->max_frame_num);

    p_mem               = mem_addr;
    p_shm               = shm_addr; 
    svr                 = (snoring_detect_t*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE); 
    p_mem               = (char*)p_mem + ALIGNED_SIZE(sizeof(snoring_detect_t), ALIGNMENT_SIZE);

    svr->fbank_memsize  = fbank_query_mem(conf->num_bins, conf->frame_length);
    svr->fbank_shmsize  = fbank_query_shm(conf->num_bins, conf->frame_length);
    svr->thinker_memsize = thinker_query_mem(conf->res_addr, conf->res_size);
    svr->thinker_shmsize = thinker_query_shm(conf->res_addr, conf->res_size);
    
    svr->p_fea          = (float*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem               = (char*)p_mem + ALIGNED_SIZE(conf->max_frame_num*conf->num_bins*sizeof(float), ALIGNMENT_SIZE);
    svr->p_classes      = (float*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem               = (char*)p_mem + ALIGNED_SIZE(conf->classes_num*sizeof(float), ALIGNMENT_SIZE);
    svr->p_fbank_mem    = (void*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem               = (char*)p_mem + ALIGNED_SIZE(svr->fbank_memsize, ALIGNMENT_SIZE);
    svr->p_thinker_mem  = (void*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem               = (char*)p_mem + ALIGNED_SIZE(svr->thinker_memsize, ALIGNMENT_SIZE);

    svr->p_fbank_shm    = (void*)ALIGNED_ADDR(p_shm, ALIGNMENT_SIZE);
    //p_shm               = (char*)p_mem + ALIGNED_SIZE(svr->fbank_shmsize, ALIGNMENT_SIZE);
    svr->p_thinker_shm  = (void*)ALIGNED_ADDR(p_shm, ALIGNMENT_SIZE);
    p_shm               = (char*)p_shm + ALIGNED_SIZE(svr->thinker_shmsize, ALIGNMENT_SIZE);

    ASSERT((uintptr_t)(p_mem) - (uintptr_t)(mem_addr) <= snoring_detect_query_mem(conf));
    ASSERT((uintptr_t)(p_shm) - (uintptr_t)(shm_addr) <= snoring_detect_query_shm(conf));

    svr->magic          = MAGIC;
    svr->p_conf         = conf;
    svr->p_fbank        = fbank_init(svr->p_fbank_mem, svr->p_fbank_shm, conf->num_bins, conf->sample_rate, conf->frame_length, conf->frame_shift);
    svr->p_thinker      = thinker_init(svr->p_thinker_mem, svr->p_thinker_shm, conf->res_addr, conf->res_size);

    return svr;
}

void snoring_detect_uninit(snoring_detect_t *svr)
{
    ASSERT(svr);
    ASSERT(MAGIC == svr->magic);
}

int snoring_detect_process(snoring_detect_t *svr, void* wave, void* classes, unsigned num_samples)
{
    snoring_detect_conf_t* p_conf;
    float *p_wave = (float*)wave;
    float *p_classes = (float*)classes;
    int ret = 0;
    unsigned num_frames, num_frames_2;

    ASSERT(svr && wave && classes);
    ASSERT(MAGIC == svr->magic);
    p_conf = svr->p_conf;
    ASSERT((num_samples >= p_conf->frame_shift) &&
        (num_samples <= p_conf->frame_length*p_conf->max_frame_num) && 
        (num_frames%p_conf->frame_shift == 0));

    num_frames = num_samples / p_conf->frame_shift;
    // compute fbank
    num_frames_2 = fbank_compute(svr->p_fbank, p_wave, svr->p_fea, num_samples);
    for (unsigned k = 0; k < num_frames; k++){
        // mean/istd
        vec_norm(svr->p_fea + k*p_conf->num_bins, svr->p_fea + k*p_conf->num_bins, p_conf->num_bins);
    }
    // backbone forward
    thinker_process(svr, p_conf, num_frames);
    // classifier
    // vec_softmaxf(svr->p_classes, svr->p_classes, p_conf->classes_num);
    memcpy(p_classes, svr->p_classes, p_conf->classes_num*sizeof(float));
    return 0;
}