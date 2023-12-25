#ifndef _LV_THINKER_H_
#define _LV_THINKER_H_

#include "thinker/thinker.h"
#include "thinker/thinker_status.h"

typedef struct tag_thinker_t thinker_t;

#pragma pack(4)
typedef struct tag_tensor_t
{
    int dim;
    int shape[4];
    int scale;
    int dtype;
    int size;
    void* data;
}tensor_t;
#pragma pack()

unsigned thinker_query_mem(void* res_addr, unsigned res_size);
unsigned thinker_query_shm(void* res_addr, unsigned res_size);
thinker_t* thinker_init(void* mem_addr, void* shm_addr, void* res_addr, unsigned res_size);
int thinker_uninit(thinker_t* svr);
int thinker_forward(thinker_t* svr);
int thinker_set_input(thinker_t* svr, int id, tensor_t *input);
int thinker_get_output(thinker_t* svr, int id, tensor_t *output);

#endif //_LV_THINKER_H_