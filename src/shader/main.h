#include "software.h"

HW(layout (local_size_x = 1, local_size_y = 1) in;)
HW(layout (rgba32f, binding = 0) uniform image2D img_output;)

HW(precision lowp    float;)

#include "shader.h"

void main () {
  ivec2 pixel_coords = ivec2 (gl_GlobalInvocationID.xy);
  pixel_coords.x *= x_batch;
  ivec2 dims = imageSize (img_output);

  float x = (float(pixel_coords.x * 2 - dims.x) / dims.x);
  float y = (float(pixel_coords.y * 2 - dims.y) / dims.y);


  vec3 xoffset = sight_x * fov;
  vec3 yoffset = -sight_y * (fov * dims.y / dims.x);
  vec3 dx = xoffset * (1.f/(dims.x / 2));
  vec3 dy = yoffset * (1.f/(dims.y / 2));

  vec3 ray = sight + xoffset * x + yoffset * y ;
  vec3 origin = viewer;
//  seed0 = frame_num;
//  seed0 = tea(floatBitsToInt(origin.x), seed0);
//  seed0 = tea(floatBitsToInt(origin.y), seed0);
//  seed0 = tea(floatBitsToInt(ray.x), seed0);
//  seed0 = tea(floatBitsToInt(ray.y), seed0);
//  seed0 = tea(floatBitsToInt(ray.z), seed0);
  seed0 = tea(pixel_coords.y * dims.x + pixel_coords.x, frame_num);


  for (int xx = 0; xx < x_batch; xx++) {
    vec3 pixel = vec3(0);
    if (frame_num != 0) {
      pixel = vec3(imageLoad (img_output, pixel_coords + ivec2(xx, 0)));
    }
    for (int i = 0; i < max_rays; i++) {
      // no normalize here to preserve focal plane
      vec3 focused_ray = (ray + dx * antialiasing(i) + dy * antialiasing(i));
      vec3 focused_point = origin + focused_ray * focused_distance;
      float r = lense_gen_r(0);
      float a = lense_gen_a(0) * 1 * M_PI;
      vec3 me = origin + sight_x * (r * cos(a)) + sight_y * (r * sin(a));
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
    imageStore (img_output, pixel_coords + ivec2(xx, 0), vec4(pixel, 1));
    ray += dx;
  }
}
