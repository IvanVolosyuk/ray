#include "software.hpp"

struct Material {
  float diffuse_ammount_;
//  float scattering_;
  float specular_exponent_;
};

struct Box {
  vec3 a_;
  vec3 b_;
};


struct Ball {
  vec3 position_;
  vec3 color_;
  float size_;
  float size2_;
  float inv_size_;
  Material material_;
};

uniform Ball balls[3]
#ifndef NO_INIT 
 = {
 { vec3(-1, -3, 0.9), vec3(1, 1, 1), 0.9, 0.9*0.9, 1/0.9,  {0.05, 500000}},
 { vec3(-3,  0, 0.9), vec3(1.00, 0.71, 0.00), 0.9, 0.9*0.9, 1/0.9, {0.00, 500000}},
 { vec3( 2,  0, 0.9), vec3(0.56, 0.56, 0.56), 0.9, 0.9*0.9, 1/0.9, {0.00, 256}}
}
#endif
;

uniform Box bbox 
#ifndef NO_INIT
 = {
  vec3(-3.9, -3.9, 0.0),
  vec3(2.9, 0.9, 1.8)
}
#endif
;

uniform Box room
#ifndef NO_INIT
= {vec3 (-60.0f, -90.0f, 0.0f ), vec3 (60.0f, 60.0f, 40.0f)}
#endif
;

