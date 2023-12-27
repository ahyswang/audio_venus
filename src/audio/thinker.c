#include "thinker.h"
#include "define.h"

typedef struct tag_thinker_t
{
    char* mem_addr;
    char* shm_addr;
    char* res_addr;
    unsigned mem_size;
    unsigned mem_used;
    unsigned shm_size;
    unsigned shm_used;
    unsigned res_size;
    tModelHandle model_hdl;
    tExecHandle hdl;
    tData inputs[8];
    tData outputs[8];
}thinker_t;

static int query_memory_plan(void* res_addr, unsigned res_size, void* mem_addr, void* shm_addr, 
    unsigned* mem_used, unsigned* shm_used)
{
    int ret = 0;
    char *p_mem, *p_shm;
    p_mem = mem_addr;
    p_shm = shm_addr;
    ret = tInitialize();
    if (ret != T_SUCCESS) {
        printf("tInitialize error, ret = %d\n", ret);
        return -1;
    }
    int num_memory = 0;
    tMemory memory_list[16];
    ret = tGetMemoryPlan((tMemory*)memory_list, &num_memory, (char*)res_addr, res_size);
    if (ret != T_SUCCESS) {
        printf("tGetMemoryPlan error, ret = %d\n", ret);
        return -1;
    }
    for (int i = 0; i < num_memory; i++) {
        int mem_size = memory_list[i].size_;
        if (0 == memory_list[i].dptr_) {
            if(1 == memory_list[i].dev_type_ || 3 == memory_list[i].dev_type_) {
                memory_list[i].dptr_ = (addr_type)(p_mem);
                p_mem = (char*)p_mem + mem_size;
            } else if (2 == memory_list[i].dev_type_) {
                memory_list[i].dptr_ = (addr_type)(p_shm);
                p_shm = (char*)p_shm + mem_size;
            }
        }
    }
    if (mem_used > 0) 
        *mem_used = (char*)p_mem - (char*)mem_addr;
    if (shm_used > 0)
        *shm_used = (char*)p_shm - (char*)shm_addr;
    return 0;
}

unsigned thinker_query_mem(void* res_addr, unsigned res_size)
{
    unsigned mem_used = 0, shm_used = 0;
    unsigned ret = ALIGNMENT_SIZE;
    ret += ALIGNED_SIZE(sizeof(thinker_t), ALIGNMENT_SIZE);
    query_memory_plan(res_addr, res_size, 0, 0, &mem_used, &shm_used);
    ret += mem_used;
    return ret;
}

unsigned thinker_query_shm(void* res_addr, unsigned res_size)
{
    unsigned mem_used = 0, shm_used = 0;
    unsigned ret = ALIGNMENT_SIZE;
    query_memory_plan(res_addr, res_size, 0, 0, &mem_used, &shm_used);
    ret += shm_used;
    return ret;
}

thinker_t* thinker_init(void* mem_addr, void* shm_addr, void* res_addr, unsigned res_size)
{
    thinker_t* svr;
    int ret = 0;
    char *p_mem, *p_shm;
    tModelHandle model_hdl;
    tExecHandle hdl;

    p_mem       = mem_addr;
    p_shm       = shm_addr;
    svr         = (thinker_t *)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_mem       = p_mem + ALIGNED_SIZE(sizeof(thinker_t), ALIGNMENT_SIZE);

    p_mem       = (char *)ALIGNED_ADDR(p_mem, ALIGNMENT_SIZE);
    p_shm       = (char *)ALIGNED_ADDR(p_shm, ALIGNMENT_SIZE);

    ret = tInitialize();
    if (ret != T_SUCCESS) {
        printf("tInitialize error, ret = %d\n", ret);
        return 0;
    }
    int num_memory = 0;
    tMemory memory_list[7];
    ret = tGetMemoryPlan((tMemory*)memory_list, &num_memory, (char*)res_addr, res_size);
    if (ret != T_SUCCESS) {
        printf("tGetMemoryPlan error, ret = %d\n", ret);
        return 0;
    }
    for (int i = 0; i < num_memory; i++) {
        int mem_size = memory_list[i].size_;
        if (0 == memory_list[i].dptr_) {
            if(1 == memory_list[i].dev_type_ || 3 == memory_list[i].dev_type_) {
                memory_list[i].dptr_ = (addr_type)(p_mem);
                p_mem = (char*)p_mem + mem_size;
            } else if (2 == memory_list[i].dev_type_) {
                memory_list[i].dptr_ = (addr_type)(p_shm);
                p_shm = (char*)p_shm + mem_size;
            }
        }
    }
    ret = tModelInit(&model_hdl, (char*)res_addr, res_size, memory_list, num_memory);
    if (ret != T_SUCCESS) {
        printf("tModelInit error, ret = %d\n", ret);
        return 0;
    }
    ret = tCreateExecutor(model_hdl, &hdl, memory_list, num_memory);
    if (ret != T_SUCCESS) {
        printf("tCreateExecutor error, ret = %d\n", ret);
        return 0;
    }
    svr->model_hdl = model_hdl;
    svr->hdl = hdl;
    return svr;
}

int thinker_uninit(thinker_t* svr)
{

    return 0;
}

int thinker_forward(thinker_t* svr)
{
    int ret = 0;
    int i;
    ret = tForward(svr->hdl);
    if (ret != T_SUCCESS) {
        printf("tForward error, ret = %d\n", ret);
        return -1;
    }
    return 0;
}


int thinker_set_input(thinker_t* svr, int id, tensor_t *input)
{
    int ret = 0;
    tData t_input;
    //t_input.dtype_ = input->dtype;
    t_input.dtype_ = Int8; //only support int8
    t_input.scale_ = input->scale;
    t_input.shape_.ndim_ = input->dim;
    for(int i = 0; i < input->dim; i++) {
        t_input.shape_.dims_[i] = input->shape[i];
    }
    t_input.dptr_ = input->data;
    ret = tSetInput(svr->hdl, id, &t_input);
    if (ret != T_SUCCESS) {
        printf("tSetInput error, ret = %d\n", ret);
        return -1;
    }
    return 0;
}

int thinker_get_output(thinker_t* svr, int id, tensor_t *output)
{
    int ret = 0;
    int size = 0;
    tData t_output;
    ret = tGetOutput(svr->hdl, id, &t_output);
    if (ret != T_SUCCESS){
        printf("tGetOutput error, ret = %d\n", ret);
        return -1;
    }
    output->dtype = t_output.dtype_;
    output->scale = t_output.scale_;
    output->dim =  (int)t_output.shape_.ndim_;
    size = 1;
    for(int i = 0; i < t_output.shape_.ndim_; i++) {
        output->shape[i] = t_output.shape_.dims_[i];
        size *= t_output.shape_.dims_[i];
    }
    if (output->size >= size) {
        memcpy(output->data, t_output.dptr_, output->size);
        output->size = size;
    } else {
        output->size = size;
    }
    return 0;
}