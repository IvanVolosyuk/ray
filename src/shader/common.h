#ifndef __COMMON_H__
#define __COMMON_H__ 1

// Common between shader.h and recursion.h
#include "software.h"

#define INPUT(type, name, value) HW(uniform) type name = value;
#include "input.h"
#undef INPUT

float fov = 0.7;
int x_batch = 8;

vec3 floor_color = vec3 (0.14, 1.0, 0.14);
vec3 wall_color = vec3 (0.85, 0.8, 0.48);
vec3 ceiling_color = vec3 (0.98, 0.98, 0.98);

struct Room {
  vec3 a_;
  vec3 b_;
} room = {vec3 (-6.0f, -6.0f, 0.0f ), vec3 (6.0f, 6.0f, 6.0f)};

float light_power = 150.4f;
vec3 light_pos = vec3(-4.2, -3, 2);
vec3 light_color = vec3(light_power, light_power, light_power);

float ball_size = 0.9f;
float ball_size2 = ball_size * ball_size;
float ball_inv_size = 1.f / ball_size;

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

struct RoomHit {
  float min_dist;
  vec3 normal;
  vec3 reflection;
  vec3 color;
};

float max_distance = 1000;
Hit no_hit = Hit(-1, max_distance, 0);

vec3 black = vec3 (0, 0, 0);

int max_internal_reflections = 30;

// Glass refraction index
float glass_refraction_index = 1.492; //1.458;

float FresnelReflectAmount (float n1, float n2, vec3 normal, vec3 incident);
Hit ball_hit(in int id, in vec3 norm_ray, in vec3 origin);
RoomHit room_hit(in vec3 norm_ray, in vec3 origin);
Hit light_hit(in vec3 norm_ray, in vec3 origin);
vec3 light_trace(in Hit p, vec3 norm_ray, vec3 origin, float distance_from_eye);

#ifdef USE_HW

float rand(float entropy) {
  seed = fract(sin(dot(vec2(seed, entropy), vec2(PHI, PI))) * SQ2);
  return seed;
}

float srand(float entropy) {
  return rand(entropy) * 2.f - 1.f;
}

vec3 wall_distr(in vec3 pos) {
  return vec3 (
      srand(pos.x) * wall_distribution,
      srand(pos.y) * wall_distribution,
      srand(pos.z) * wall_distribution);
}

float uniform_rand(float entroy) {
  float x = rand(entropy);
  return return 1/x + 1/(x-1);
}

vec3 light_distr(in vec3 point) {
  return vec3(
      uniform_rand(point.z) * light_size,
      uniform_rand(point.x) * light_size,
      uniform_rand(point.y) * light_size);
}

float lense_gen(in float a) {
  return srand(a) * lense_blur;
}

float antialiasing(in float c) {
  return srand(c) * 0.5;
}

float reflect_gen(in vec3 point) {
  return rand(point.x * point.y * point.z);
}

#else  // SW
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> lense_gen{-lense_blur,lense_blur};
std::uniform_real_distribution<float> reflect_gen{0.f, 1.f};

std::uniform_real_distribution<float> wall_gen{0, 1};
std::uniform_real_distribution<float> light_gen{-light_size, light_size};
std::uniform_real_distribution<float> antialiasing{-0.5,0.5};
vec3 wall_distr() {
  auto d = [](float x) { return 1/x + 1/(x-1); };
  return vec3(
      d(wall_gen(gen)) * wall_distribution,
      d(wall_gen(gen)) * wall_distribution,
      d(wall_gen(gen))* wall_distribution);
}


vec3 light_distr() {
    return vec3(light_gen(gen), light_gen(gen), light_gen(gen));
}

#endif  // USE_HW

#endif
