#ifndef __SOFTWARE_H__
#define __SOFTWARE_H__ 1

#define MAX_FLOAT 1e37

#ifdef USE_HW
// Hardware mode

#define TEXTURE(N, name)                               \
layout(std430, binding = N) buffer name {              \
  int width_##N;                                       \
  int height_##N;                                      \
  float specular_exponent_##N;                         \
  float diffuse_ammount_##N;                           \
  highp uint pixels_##N[];                             \
};

#define HW(x) x
#define SW(x)
#define LENGTH(a) a.length()
#define size_t int
#define assert(x)
#define isfinite(x) (!isnan(x))
#define M_PI 3.14159265358979323846264
#define swap(a,b) { float x = a; a = b; b = x; }

#else  // not USE_HW

// Software mode
#include <cmath>
#include <functional>
#include <algorithm>
#include <random>
#include <utility>
#include "vector.hpp"
#include "texture.hpp"

inline float sign(float a) { return std::copysign(1, a); }
inline float inversesqrt(float a) { return 1./sqrtf(a); }

#define HW(x)
#define SW(x) x

#define sqrt(x) sqrtf(x)
#define abs(x) fabs(x)
#define in const
#define Hit(a,b,c) Hit{a,b,c}
#define RoomHit(a,b,c,d) RoomHit{a,b,c,d}
#define SineHit(a,b,c) SineHit{a,b,c}
#define LENGTH(a) (sizeof(a)/sizeof(a[0]))
using std::max;
using std::min;
using std::clamp;
using std::swap;

#endif  // USE_HW

#endif  // __SOFTWARE_H__
