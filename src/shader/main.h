layout (local_size_x = 1, local_size_y = 1) in;
layout (rgba32f, binding = 0) uniform image2D img_output;

uniform vec3 viewer;
uniform vec3 sight;
uniform float focused_distance;

struct Room {
  vec3 a_;
  vec3 b_;
} room = {vec3 (-6.0f, -6.0f, 0.0f ), vec3 (6.0f, 6.0f, 6.0f)};

vec3 black = vec3 (0, 0, 0);
vec3 floor_color = vec3 (0.14, 1.0, 0.14);
vec3 wall_color = vec3 (0.85, 0.8, 0.48);
vec3 ceiling_color = vec3 (0.98, 0.98, 0.98);

vec3 light_pos = vec3(-4.2, -3, 2);
float light_size = 0.9f;
float light_power = 150.4f;
vec3 light_color = vec3(light_power, light_power, light_power);
float light_size2 = light_size * light_size;
float light_inv_size = 1.f / light_size;

float ball_size = 0.9f;
float ball_size2 = ball_size * ball_size;
float ball_inv_size = 1.f / ball_size;

float defuse_attenuation = 0.4;
float max_distance = 1000;

// vec3 viewer = vec3 (0f, -5.5f, 1.5f);
// vec3 sight = normalize(vec3(0.0, 1.0, -0.1));
// float focused_distance = 3.1;

int max_rays = 32;
int max_internal_reflections = 30;


struct Ball {
  vec3 position_;
  vec3 color_;
} balls[3] = {
 { vec3(-1, -2, ball_size * 1.0f), vec3(1, 1, 1)},
 { vec3(-2 * ball_size, 0, ball_size), vec3(0.01, 1.0, 0.01)},
 { vec3(2 * ball_size, 0, ball_size), vec3(1.00, 0.00, 1.)}
};

struct Hit {
  int id_;
  float closest_point_distance_from_viewer_;
  float distance_from_object_center2_;
};
Hit no_hit = Hit(-1, max_distance, 0);

precision lowp    float;

float PHI = 1.61803398874989484820459 * 00000.1; // Golden Ratio   
float PI  = 3.14159265358979323846264 * 00000.1; // PI
float SQ2 = 1.41421356237309504880169 * 10000.0; // Square Root of Two

float seed = PI;
//vec2 noise_pos;

//float gold_noise(in vec2 coordinate, in float seed){
//      return fract(sin(dot(coordinate*(seed+PHI), vec2(PHI, PI)))*SQ2);
//}

float rand(float entropy) {
  seed = fract(sin(dot(vec2(seed, entropy), vec2(PHI, PI))) * SQ2);
  return seed;
}

//float rand(float entropy) {
//  seed = gold_noise(noise_pos, seed + entropy);
//  return seed;
//}

float srand(float entropy) {
  return (rand(entropy) * 2.f - 1.f);
}

// IMPL

// FIXME: normal distribution
vec3 wall_distr(in vec3 pos) {
  return vec3 (
      srand(pos.x) * 0.8,
      srand(pos.y) * 0.8,
      srand(pos.z) * 0.8);
}
vec3 light_distr(in vec3 point) {
  return vec3(
      srand(point.z) * light_size,
      srand(point.x) * light_size,
      srand(point.y) * light_size);
}

float lense_gen(in float a) {
  return srand(a) * 0.03;
}

float antialiasing(in float c) {
  return srand(c);
}

float reflect_gen(in vec3 point) {
  return rand(point.x * point.y * point.z);
}

// Glass refraction index
float glass_refraction_index = 1.492; //1.458;
//float OBJECT_REFLECTIVITY = 0;

float FresnelReflectAmount (float n1, float n2, vec3 normal, vec3 incident) {
  // Schlick aproximation
  float r0 = (n1-n2) / (n1+n2);
  r0 *= r0;
  float cosX = -dot(normal, incident);
  if (n1 > n2)
  {
    float n = n1/n2;
    float sinT2 = n*n*(1.0-cosX*cosX);
    // Total internal reflection
    if (sinT2 > 1.0)
      return 1.0;
    cosX = sqrt(1.0-sinT2);
  }
  float x = 1.0-cosX;
  float ret = r0+(1.0-r0)*x*x*x*x*x;

  // adjust reflect multiplier for object reflectivity
//  ret = (OBJECT_REFLECTIVITY + (1.0-OBJECT_REFLECTIVITY) * ret);
  return ret;
}

Hit ball_hit(in int id, in vec3 norm_ray, in vec3 origin) {
  vec3 ball_vector = balls[id].position_ - origin;

  float closest_point_distance_from_viewer = dot(norm_ray, ball_vector);
  if (closest_point_distance_from_viewer < 0) {
    return no_hit;
  }

  float ball_distance2 = dot(ball_vector, ball_vector);
  float distance_from_object_center2 = ball_distance2 -
    closest_point_distance_from_viewer * closest_point_distance_from_viewer;
  if (distance_from_object_center2 > ball_size2) {
    return no_hit;
  }
  return Hit(id, closest_point_distance_from_viewer, distance_from_object_center2);
}

struct RoomHit {
  float min_dist;
  vec3 normal;
  vec3 reflection;
  vec3 color;
};

RoomHit room_hit(in vec3 norm_ray, in vec3 origin) {
  vec3 tMin = (room.a_ - origin) / norm_ray;
  vec3 tMax = (room.b_ - origin) / norm_ray;
  vec3 t1 = min(tMin, tMax);
  vec3 t2 = max(tMin, tMax);
//  float tNear = max(max(t1.x, t1.y), t1.z);
//  float tFar = min(min(t2.x, t2.y), t2.z);

  vec3 normal;
  vec3 reflection;
  float min_dist;
  vec3 color;

  if (t2.y < t2.z) {
    color = wall_color;
    if (t2.x < t2.y) {
      min_dist = t2.x;
      reflection = vec3(-norm_ray.x, norm_ray.y, norm_ray.z);
      normal = vec3(sign(reflection.x), 0, 0);
    } else {
      min_dist = t2.y;
      reflection = vec3(norm_ray.x, -norm_ray.y, norm_ray.z);
      normal = vec3(0, sign(reflection.y), 0);
    }
  } else {
    if (t2.x < t2.z) {
      min_dist = t2.x;
      reflection = vec3(-norm_ray.x, norm_ray.y, norm_ray.z);
      normal = vec3(sign(reflection.x), 0, 0);
      color = wall_color;
    } else {
      min_dist = t2.z;
      reflection = vec3(norm_ray.x, norm_ray.y, -norm_ray.z);
      normal = vec3(0, 0, sign(reflection.z));
      color = reflection.z < 0 ? ceiling_color : floor_color;
    }
  }
  return RoomHit(min_dist, normal, reflection, color);
}

Hit light_hit(in vec3 norm_ray, in vec3 origin) {
  vec3 light_vector = light_pos - origin;
  float light_distance2 = dot(light_vector, light_vector);

  float closest_point_distance_from_origin = dot(norm_ray, light_vector);
  if (closest_point_distance_from_origin < 0) {
    return no_hit;
  }

  float distance_from_light_center2 = light_distance2 -
    closest_point_distance_from_origin * closest_point_distance_from_origin;
  if (distance_from_light_center2 > light_size2) {
    return no_hit;
  }
  return Hit(-2, closest_point_distance_from_origin, distance_from_light_center2);
}

vec3 light_trace(
    in Hit p,
    vec3 norm_ray,
    vec3 origin,
    float distance_from_eye) {
  float distance_from_origin = p.closest_point_distance_from_viewer_ -
    sqrt(ball_size2 - p.distance_from_object_center2_);

  vec3 intersection = origin + norm_ray * distance_from_origin;
  vec3 distance_from_light_vector = intersection - light_pos;

  vec3 normal = distance_from_light_vector * light_inv_size;
  float angle = -dot(norm_ray, normal);
  float total_distance = distance_from_eye + distance_from_origin;

  return light_color * (angle / (total_distance * total_distance));
}

#include "stages.h"

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
