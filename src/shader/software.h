#ifndef __SOFTWARE_H__
#define __SOFTWARE_H__ 1

#ifdef USE_HW
// Hardware mode

#define HW(x) x
#define SW(x)
#define LENGTH(a) a.length()
#define size_t int
#else  // not USE_HW

// Software mode
#include "vector.hpp"
#include <cmath>
#include <functional>
#include <random>

float sign(float a) {
  return std::copysign(1, a);
}
float inversesqrt(float a) { return 1./sqrtf(a); }

#define HW(x)
#define SW(x) x

#define sqrt(x) sqrtf(x)
#define in const
#define Hit(a,b,c) Hit{a,b,c}
#define RoomHit(a,b,c,d) RoomHit{a,b,c,d}
#define LENGTH(a) (sizeof(a)/sizeof(a[0]))

#endif  // USE_HW

#endif  // __SOFTWARE_H__
