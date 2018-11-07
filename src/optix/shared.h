#define RT_USE_TEMPLATED_RTCALLABLEPROGRAM 1

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#ifndef RT_FUNCTION
#define RT_FUNCTION __forceinline__ __device__
#endif


class vec3 {
  public:
  union {
    struct {
      float x;
      float y;
      float z;
    };
    float a[3];
#ifdef CUDA
    float3 f3;
    vec3(float3 v) : f3(v) {}
#endif
  };
};

