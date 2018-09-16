#ifndef INPUT
#include "software.h"
#define INPUT(type, name, value) extern type name;
#define DEFAULT_INPUT 1
#endif

INPUT(vec3, sight_y, vec3(0,0,1))
INPUT(vec3, sight_x, vec3(1,0,0))
INPUT(vec3, viewer, vec3(0, -5.5, 1.5))
INPUT(vec3, sight, normalize(vec3(0., 1, -0.1)))
INPUT(float, focused_distance, 3.1)
INPUT(float, light_size, 0.9f)
INPUT(float, light_size2, 0.8f) // light_size * light_size
//INPUT(float, light_inv_size, 1.1f) // 1.f / light_size
INPUT(int, frame_num, 0) // 1.f / light_size
//INPUT(float, diffuse_attenuation, 0.4)
INPUT(int, max_depth, 2)
INPUT(float, lense_blur, 0.01)
INPUT(int, max_rays, 1)




#ifdef DEFAULT_INPUT
#undef DEFAULT_INPUT
#undef INPUT
#endif
