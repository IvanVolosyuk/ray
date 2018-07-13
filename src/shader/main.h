#include "software.h"

HW(layout (local_size_x = 1, local_size_y = 1) in;)
HW(layout (rgba32f, binding = 0) uniform image2D img_output;)

// vec3 viewer = vec3 (0f, -5.5f, 1.5f);
// vec3 sight = normalize(vec3(0.0, 1.0, -0.1));
// float focused_distance = 3.1;

int max_rays = 1;


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
  rand(x);
  rand(y);

  float dx = 0.95;
  vec3 sight_x = normalize(cross(sight, vec3(0,0,1)));
  vec3 sight_y = -cross(sight, sight_x) * dims.y / dims.x;
  sight_x *= dx;
  sight_y *= dx;

  vec3 ray = sight + sight_x * x + sight_y * y ;
  vec3 origin = viewer;
  vec3 norm_ray = normalize(ray);
  vec3 aax = sight_x / dims.x * 2;
  vec3 aay = sight_y / dims.y * 2;

  vec3 pixel = vec3 (0.0, 0.0, 0.0);

  for (int i = 0; i < max_rays; i++) {
    vec3 focused_ray = normalize(norm_ray + aax * antialiasing(i) + aay * antialiasing(i));
    vec3 focused_point = origin + focused_ray * focused_distance;
    vec3 me = origin + sight_x * lense_gen(x * 0.0123 + y * 0.07543 + i * 0.12)
                     + sight_y * lense_gen(x * 0.0652 + y * 0.022571 + i * 0.77);
    vec3 new_ray = normalize(focused_point - me);

    pixel += trace_2(new_ray, me, 0.f);
  }
  pixel /= max_rays;

  imageStore (img_output, pixel_coords, vec4(pixel, 1));
}
