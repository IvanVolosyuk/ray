#define CURR(n) n ## _0
#include "recursion.h"
#undef CURR

#define CURR(n) n ## _1
#define NEXT(n) n ## _0
#include "recursion.h"
#undef NEXT
#undef CURR

#define CURR(n) n ## _2
#define NEXT(n) n ## _1
#include "recursion.h"
#undef NEXT
#undef CURR

#define MAX_STAGE 1
#define CURR(n) n ## _3
#define NEXT(n) n ## _2
#include "recursion.h"
#undef NEXT
#undef CURR

//#define CURR(n) n ## _4
//#define NEXT(n) n ## _3
//#include "recursion.h"
//#undef NEXT
//#undef CURR
