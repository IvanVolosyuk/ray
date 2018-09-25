#ifndef __COMMON_H__
#define __COMMON_H__ 1

// Common between shader.h and recursion.h
#include "software.hpp"

#define INPUT(type, name, value) HW(uniform) type name = value;
#include "input.hpp"
#undef INPUT

#include "struct_input.hpp"

float fov = 0.7;
int x_batch = 8;

float light_power = 200.4f;
vec3 light_pos = vec3(5.0, -8, 3.0);
vec3 light_color = vec3(light_power, light_power, light_power);

struct Hit {
  int id_;
  float closest_point_distance_from_viewer_;
  float distance_from_object_center2_;
};

struct RoomHit {
  float min_dist;
  vec3 intersection;
  vec3 normal;
  vec3 reflection;
  vec3 color;
  Material material;
};

float max_distance = 1000;
Hit no_hit = Hit(-1, max_distance, 0);

vec3 black = vec3 (0, 0, 0);

vec3 refract(float ior, vec3 N, vec3 I);
float fresnel(float ior, vec3 N, vec3 I);
Hit ball_hit(in int id, in vec3 norm_ray, in vec3 origin);
RoomHit room_hit(in vec3 norm_ray, in vec3 origin);
Hit light_hit(in vec3 norm_ray, in vec3 origin);
vec3 light_trace(in Hit p, vec3 norm_ray, vec3 origin, float distance_from_eye);
vec3 sine_trace(in Hit p, vec3 norm_ray, vec3 origin, float distance_from_eye);
vec3 scatter(in vec3 v, float specular_exponent);

#ifdef USE_HW

uint seed0 = 2;
uint seq = 0;
uint k0 = 0xA341316C;
uint k1 = 0xC8013EA4;
uint k2 = 0xAD90777D;
uint k3 = 0x7E95761E;
uint sum = 0x9e3779b9;

uint tea(uint v0, uint v1) {
  v0 +=((v1 << 4)+k0) ^ (v1 + sum) ^ ((v1 >> 5)+k1);
  v1 +=((v0 << 4)+k2) ^ (v0 + sum) ^ ((v0 >> 5)+k3);
  v0 +=((v1 << 4)+k0) ^ (v1 + sum) ^ ((v1 >> 5)+k1);
  v1 +=((v0 << 4)+k2) ^ (v0 + sum) ^ ((v0 >> 5)+k3);
  v0 +=((v1 << 4)+k0) ^ (v1 + sum) ^ ((v1 >> 5)+k1);
  v1 +=((v0 << 4)+k2) ^ (v0 + sum) ^ ((v0 >> 5)+k3);
  v0 +=((v1 << 4)+k0) ^ (v1 + sum) ^ ((v1 >> 5)+k1);
  v1 +=((v0 << 4)+k2) ^ (v0 + sum) ^ ((v0 >> 5)+k3);
  return v0;
}

// https://en.wikipedia.org/wiki/Linear_congruential_generator
uint LCG_mul = 1664525;
uint LCG_incr = 1013904223;
uint LCG_mask = 0xFFFFFF; // 24 lower bits
float LCG_normalizer = 5.960464477539063e-08f;//1.f/(float)LCG_mask;

float rand() {
  seed0 = seed0 * LCG_mul + LCG_incr;
  return (seed0 & LCG_mask) * LCG_normalizer;
}

vec2 rand2() {
  return vec2(rand(), rand());
}

float srand() {
  return rand() * 2.f - 1.f;
}


vec2 normal_rand() {
  vec2 rr = rand2();
  if (rr.x == 0) return vec2(0,0);
  float r = sqrt(-2 * log(rr.x));
  float a = rr.y * 2 * M_PI;

  return vec2 (cos(a) * r, sin(a) * r);
}

vec3 wall_distr(float scattering) {
  vec2 n1 = normal_rand();
  vec2 n2 = normal_rand();
  return vec3 (
      n1.x * scattering,
      n2.x * scattering,
      n2.y * scattering);
}

vec3 light_distr() {
  while (true) {
    float x = srand();
    float y = srand();
    float z = srand();
    if (x*x + y*y + z*z <= 1) {
      return vec3(x,y,z) * light_size;
    }
  }
}

float lense_gen_r(in float a) {
  return sqrt(rand()) * lense_blur;
}

float lense_gen_a(in float a) {
  return rand() * 2 * M_PI;
}

float antialiasing(in float c) {
  return srand() * 0.5;
}

float reflect_gen() {
  return rand();
}

#else  // SW
std::random_device rd;
std::mt19937 gen(rd());
// FIXME: use tee for seed and Linear congruential generator for intermidiate randoms
// FIXME: use aabb fast ray intersection, construct nested bounding boxes
std::uniform_real_distribution<float> lense_gen_r{0,1};
std::uniform_real_distribution<float> lense_gen_a{0,2 * M_PI};
std::uniform_real_distribution<float> reflect_gen{0.f, 1.f};

std::uniform_real_distribution<float> light_gen{-1, 1};
std::uniform_real_distribution<float> antialiasing{-0.5,0.5};

vec3 light_distr() {
  while (true) {
    float x = light_gen(gen);
    float y = light_gen(gen);
    float z = light_gen(gen);
    if (x*x + y*y + z*z <= 1) {
      return vec3(x,y,z) * light_size;
    }
  }
}

#endif  // USE_HW

TEXTURE(0, floor_tex)
TEXTURE(1, wall_tex)
TEXTURE(2, ceiling_tex)


#endif
