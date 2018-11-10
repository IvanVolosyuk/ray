#define RT_USE_TEMPLATED_RTCALLABLEPROGRAM 1

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#ifndef RT_FUNCTION
#define RT_FUNCTION __forceinline__ __device__
#endif

class vec3 {
  public:
    vec3() = default;
    RT_FUNCTION
    vec3(float3 v) : f3(v) {}
    RT_FUNCTION
    operator float3() const { return f3; }
    RT_FUNCTION
    operator float3&() { return f3; }
    RT_FUNCTION
    float operator [](int idx) const { return a[idx]; }

    RT_FUNCTION
    vec3(float x, float y, float z) : x(x), y(y), z(z) {}
  union {
    struct {
      float x;
      float y;
      float z;
    };
    float a[3];
#ifdef rtBuffer
    float3 f3;
#endif
  };
};

struct AABB {
  AABB() = default;
  vec3 min;
  vec3 max;
};

struct tri {
  vec3 vertex[3];
  vec3 normal;
  vec3 vertex_normal[3];
};

struct kd {
  public:
    kd() = default;
    int split_axe_and_idx;
    float split_line;
};

#define P(a) {}
#define likely(a) (a)
#define unlikely(a) (a)
