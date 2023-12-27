#include <stdio.h>
#include <stdlib.h>
#include "audio/define.h"
#include "snoring_detect/snoring_detect.h"
#include "snoring_net.quant.pkg.h"
#include "audio_segment_20_pad_float32.bin.h"

#define MEM_SIZE    (1*1024*1024)
#define SHM_SIZE    (640*1024)

#define LOG printf
#define LOGD printf

#if defined(linux)
static char g_mem_buf[MEM_SIZE] __attribute__((aligned(32)));
static char g_shm_buf[SHM_SIZE] __attribute__((aligned(32)));
unsigned g_mem_size = MEM_SIZE;
unsigned g_shm_size = SHM_SIZE;
void* g_mem_addr = g_mem_buf;
void* g_shm_addr = g_shm_buf;
#else 
static char g_mem_buf[MEM_SIZE] __attribute__((aligned(32)));
unsigned g_mem_size = MEM_SIZE;
unsigned g_shm_size = SHM_SIZE;
void* g_mem_addr = g_mem_buf;
void* g_shm_addr = 0x5fe00000;
#endif 

int load_bin(const char* filename, void** pp_addr, unsigned* p_size)
{
#if defined(linux)
    FILE* fp;
    unsigned size;
    void* buf;
    fp = fopen(filename, "rb");
    if (!fp) {
        printf("open file error, file:%s\n", filename);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    buf = malloc(size);
    fread(buf, size, 1, fp);
    fclose(fp);
    *pp_addr = buf;
    *p_size = size;
#endif
    return 0;
}

void main_app(int argc, char **argv)
{
    void *p_mem, *p_shm;
    snoring_detect_t *kws;
    snoring_detect_conf_t conf;
    
    int num_frames = 0;
    int num_bins = 64;// 80 dim fbank
    int sample_rate = 16000;// 16k sample rate
    int frame_length = sample_rate / 1000 * 64;// frame length 25ms
    int frame_shift = sample_rate / 1000 * 32;// frame shift 10ms
    int keyword_num = 2;
    int max_frame_num = 64;
    long long last_counter;
    
    void *mem_addr, *shm_addr;
    unsigned mem_size, shm_size;
    float *wave, *fea;

    void *res_addr, *input_addr, *meanstd_addr;
    unsigned res_size, input_size, meanstd_size;

    LOG("func:%s\n", __FUNCTION__);
#if 0
    const char* res_path = argv[1];
    const char* wav_path = argv[2];
    load_bin(res_path, &res_addr, &res_size);
    load_bin(wav_path, &input_addr, &input_size);
#else 
    res_addr = snoring_net_quant_pkg_resource;
    res_size = snoring_net_quant_pkg_reslen;
    input_addr = audio_segment_20_pad_float32_resource;
    input_size = audio_segment_20_pad_float32_reslen;
#endif

    LOG("res_size:%d\n", res_size);
    LOG("input_size:%d\n", input_size);

    memset(&conf, 0, sizeof(conf));
    conf.num_bins = num_bins;
    conf.sample_rate = sample_rate;
    conf.frame_length = frame_length;
    conf.frame_shift = frame_shift;
    conf.classes_num = keyword_num;
    conf.max_frame_num = 64;
    conf.qvalue = 4; // from onnx quant scale
    conf.res_addr = res_addr; 
    conf.res_size = res_size;
    conf.extra1 = 0;
    conf.extra2 = 0;

    mem_size = snoring_detect_query_mem(&conf);
    shm_size = snoring_detect_query_shm(&conf);
    printf("mem_size:%d, shm_size:%d\n", mem_size, shm_size);

    p_mem       = g_mem_addr;
    mem_addr    = (void*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem       = (char*)p_mem + ALIGNED_SIZE(mem_size, ALIGNMENT_SIZE);
    wave        = (void*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem       = (char*)p_mem + ALIGNED_SIZE(max_frame_num*frame_shift*sizeof(float), ALIGNMENT_SIZE);
    fea         = (void*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem       = (char*)p_mem + ALIGNED_SIZE(keyword_num*sizeof(float), ALIGNMENT_SIZE);
    
    p_shm       = g_shm_addr;
    shm_addr    = (void*)ALIGNED_ADDR(p_shm, ALIGNMENT_SIZE);
    p_shm       = (char*)p_shm + ALIGNED_SIZE(shm_size, ALIGNMENT_SIZE);

    ASSERT((uintptr_t)(p_mem) - (uintptr_t)(g_mem_addr) <= g_mem_size);
    ASSERT((uintptr_t)(p_shm) - (uintptr_t)(g_shm_addr) <= g_shm_size);

    kws = snoring_detect_init(mem_addr, shm_addr, &conf);
    ASSERT(kws != 0);

    num_frames = input_size / sizeof(float);
    if (num_frames > max_frame_num*frame_shift) {
        num_frames = max_frame_num*frame_shift;
    }

    for (int j = 0; j < num_frames; j++) {
        wave[j] = ((float*)(input_addr))[j];
    }
    last_counter = clock();
    snoring_detect_process(kws, wave, fea, num_frames);
    last_counter = clock() - last_counter;
    for (int j = 0; j < keyword_num; j++) {
        printf("keyword:%d, pred:%f\n", j, fea[j]);
    }
    printf("counter:%f(M/s)\n", last_counter/1.0e6f/2.0);

    snoring_detect_uninit(kws);
}

int main(int argc, char** argv)
{
    main_app(argc, argv);
    return 0;
}