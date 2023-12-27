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
#endif 
    return 0;
}

int save_bin(const char* filename, void* p_addr, unsigned size)
{
#if defined(linux)
    FILE* fp;
    void* buf;
    fp = fopen(filename, "wb");
    if (!fp) {
        printf("open file error, file:%s\n", filename);
        return -1;
    }
    fwrite(buf, size, 1, fp);
    fclose(fp);
#endif 
    return 0;
}

