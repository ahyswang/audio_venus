#include "test.h"

static char g_mem_buf[MEM_SIZE] __attribute__((aligned(32)));
static char g_shm_buf[SHM_SIZE] __attribute__((aligned(32)));
unsigned g_mem_size = MEM_SIZE;
unsigned g_shm_size = SHM_SIZE;
void* g_mem_addr;
void* g_shm_addr;

int test_init()
{
    g_mem_size = MEM_SIZE;
    g_shm_size = SHM_SIZE;
    g_mem_addr = g_mem_buf;
    g_shm_addr = g_shm_buf;
}

void test_uninit()
{

}