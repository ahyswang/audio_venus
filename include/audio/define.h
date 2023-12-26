#ifndef _LV_DEFINE_H_
#define _LV_DEFINE_H_

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef true
#define true 1
#endif 
#ifndef false
#define false 0
#endif 
#ifndef size_t 
#define size_t unsigned long
#endif

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

#endif //_LV_DEFINE_H_