#include "mylib.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int myadd(int a, int b)
{
    return a + b;
}

#define LOG printf 
#define ASSERT(c) \
if (!(c)) { \
    LOG("%s (#%d): assert(%s\n)", __FILE__, __LINE__, #c); \
    exit(1); \
}
#define ALIGNMENT_SIZE (8)
#define ALIGNED_SIZE(size, align) \
    ((size_t)(size)+(align)-1)
#define ALIGNED_ADDR(addr, align) \
    (((uintptr_t)(addr)+((align)-1ul)) & (~((align)-1ul)))
#define ALIGNE(size, align) \
    (((size_t)(size)+((align)-1ul)) & (~((align)-1ul)))

typedef struct mylib_t_
{
    void* p_res;
    unsigned res_size;
}mylib_t;

unsigned mylib_query_mem(mylib_conf_t* conf)
{
    uint32_t ret = 0;
    ASSERT(conf);
    ASSERT(conf->p_res[0] > 0 && conf->p_size[0] > 0);

    LOG("conf = %p\n");
    LOG("conf->p_res[0] = %p\n", conf->p_res[0]);
    LOG("conf->p_size[0] = %d\n", conf->p_size[0]);

    ret = ALIGNMENT_SIZE;
    ret += ALIGNED_SIZE(sizeof(mylib_t), ALIGNMENT_SIZE);
    ret += ALIGNED_SIZE(conf->p_size[0], ALIGNMENT_SIZE);
    return ret;
}

mylib_t* mylib_init(void* mem_addr, mylib_conf_t* conf)
{
    mylib_t* p_this;
    void *p_mem;
    int ret;

    ASSERT(conf);
    ASSERT(conf->p_res[0] > 0 && conf->p_size[0] > 0);

    LOG("mem_addr = %p\n", mem_addr);
    LOG("conf = %p\n", conf);
    LOG("conf->p_res[0] = %p\n", conf->p_res[0]);
    LOG("conf->p_size[0] = %d\n", conf->p_size[0]);

    p_mem           = mem_addr;
    p_this          = (mylib_t*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem           = (char*)p_mem + ALIGNED_SIZE(sizeof(mylib_t), ALIGNMENT_SIZE);
    p_this->p_res   = (void*)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem           = (char*)p_mem + ALIGNED_SIZE(sizeof(conf->p_size[0]), ALIGNMENT_SIZE);

    p_this->res_size = conf->p_size[0];
    memcpy(p_this->p_res, conf->p_res[0], conf->p_size[0]);

    ASSERT((uintptr_t)(p_mem) - (uintptr_t)(mem_addr) <= mylib_query_mem(conf));

    LOG("p_this = %p\n", p_this);

    return p_this;
}

void mylib_uninit(mylib_t* p_this)
{

}

int mylib_process(mylib_t* p_this, void* input, void* output, unsigned int size)
{
    ASSERT(p_this);
    ASSERT(input);
    ASSERT(output);
    ASSERT(size > 0);

    LOG("p_this = %p, input = %p, output = %p, size = %d\n", p_this, input, output, size);

    memcpy(output, p_this->p_res, p_this->res_size);
    memcpy(output + p_this->res_size, input, size);

    return 0;
}
