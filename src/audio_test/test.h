#ifndef _LV_TEST_H_
#define _LV_TEST_H_

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef ASSERT
#define ASSERT(c) \
if (!(c)) { \
    printf("%s (#%d): assert(%s\n)", __FILE__, __LINE__, #c); \
    exit(1); \
}
#endif 

#ifndef ALIGNMENT_SIZE
#define ALIGNMENT_SIZE (8)
#define ALIGNED_SIZE(size, align) \
    ((size_t)(size)+(align)-1)
#define ALIGNED_ADDR(addr, align) \
    (((uintptr_t)(addr)+((align)-1ul)) & (~((align)-1ul)))
#define ALIGNE(size, align) \
    (((size_t)(size)+((align)-1ul)) & (~((align)-1ul)))
#endif 

#define MEM_SIZE    (1*1024*1024)
#define SHM_SIZE    (640*1024)

#define LOG printf
#define LOGD printf

#ifdef __cplusplus
extern "C"
{
#endif

extern unsigned g_mem_size;
extern unsigned g_shm_size;
extern void* g_mem_addr;
extern void* g_shm_addr;

int test_init();
void test_uninit();

// test unit
void test_fbank();
void test_fft();

int load_bin(const char* filename, void** pp_addr, unsigned* p_size);
int save_bin(const char* filename, void* p_addr, unsigned size);

#ifdef __cplusplus
}
#endif

#endif // _LV_TEST_H_