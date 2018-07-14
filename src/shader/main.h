#include "software.h"

HW(layout (local_size_x = 1, local_size_y = 1) in;)
HW(layout (rgba32f, binding = 0) uniform image2D img_output;)

HW(precision lowp    float;)

float PHI = 1.61803398874989484820459 * 00000.1; // Golden Ratio   
float PI  = 3.14159265358979323846264 * 00000.1; // PI
float SQ2 = 1.41421356237309504880169 * 10000.0; // Square Root of Two

float seed = PI;

#include "shader.h"

void main () {
  ivec2 pixel_coords = ivec2 (gl_GlobalInvocationID.xy);
  ivec2 dims = imageSize (img_output);

  float x = (float(pixel_coords.x * 2 - dims.x) / dims.x);
  float y = (float(pixel_coords.y * 2 - dims.y) / dims.y);
  seed = frame_num + x * 10.01231 + y * 17.234231;

  vec3 xoffset = sight_x * fov;
  vec3 yoffset = -sight_y * (fov * dims.y / dims.x);
  vec3 dx = xoffset * (1.f/(dims.x / 2));
  vec3 dy = yoffset * (1.f/(dims.y / 2));

  vec3 ray = sight + xoffset * x + yoffset * y ;
  vec3 origin = viewer;

  vec3 pixel = vec3(0);
  if (frame_num != 0) {
    pixel = vec3(imageLoad (img_output, pixel_coords));
  }

  for (int i = 0; i < max_rays; i++) {
    vec3 focused_ray = normalize(ray + dx * antialiasing(i) + dy * antialiasing(i));
    vec3 focused_point = origin + focused_ray * focused_distance;
    vec3 me = origin + sight_x * lense_gen(x * 0.0123 + y * 0.07543 + i * 0.12)
                     + sight_y * lense_gen(x * 0.0652 + y * 0.022571 + i * 0.77);
    vec3 new_ray = normalize(focused_point - me);
    if (max_depth > 1) {
      if (max_depth > 2) {
        pixel += trace_3(new_ray, me, 0.f);
      } else {
        pixel += trace_2(new_ray, me, 0.f);
      }
    } else {
      if (max_depth == 0) {
        pixel += trace_0(new_ray, me, 0.f);
      } else {
        pixel += trace_1(new_ray, me, 0.f);
      }
    }
  }

  imageStore (img_output, pixel_coords, vec4(pixel, 1));
}
