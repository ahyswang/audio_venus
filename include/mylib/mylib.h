#ifndef _MYLIB_H_
#define _MYLIB_H_

int myadd(int a, int b);


typedef struct mylib_conf_t_
{
    char* p_res[4];
    unsigned p_size[4];
} mylib_conf_t;
typedef struct mylib_t_ mylib_t;

unsigned mylib_query_mem(mylib_conf_t* conf);
mylib_t* mylib_init(void* mem_addr, mylib_conf_t* conf);
void mylib_uninit(mylib_t* p_this);
int mylib_process(mylib_t* p_this, void* input, void* output, unsigned int);


#endif