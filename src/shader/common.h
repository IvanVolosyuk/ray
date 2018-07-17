#ifndef __COMMON_H__
#define __COMMON_H__ 1

// Common between shader.h and recursion.h
#include "software.h"

#define INPUT(type, name, value) HW(uniform) type name = value;
#include "input.h"
#undef INPUT

float PI = 3.14159265358979323846264;

float fov = 0.7;
int x_batch = 8;

vec3 floor_color = vec3 (0.14, 1.0, 0.14);
vec3 wall_color = vec3 (0.85, 0.8, 0.48);
vec3 ceiling_color = vec3 (0.98, 0.98, 0.98);

struct Box {
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
 { vec3(-1, -2, ball_size), vec3(1, 1, 1)},
 { vec3(-2 * ball_size, 0, ball_size), vec3(0.01, 1.0, 0.01)},
 { vec3(2 * ball_size, 0, ball_size), vec3(1.00, 0.00, 1.)}
};

Box bbox = {
  // precomputed
  vec3(-2.7, -2.9, 0.0),
  vec3(2.7, 0.9, 1.8)
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

uint seed0 = 2;
uint seq = 0;
uint k0 = 0xA341316C;
uint k1 = 0xC8013EA4;
uint k2 = 0xAD90777D;
uint k3 = 0x7E95761E;
uint sum = 0x9e3779b9;

ivec2 irand2() {
  uint v0 = seed0;
  uint v1 = seq++;
  v0 +=((v1 << 4)+k0) ^ (v1 + sum) ^ ((v1 >> 5)+k1);
  v1 +=((v0 << 4)+k2) ^ (v0 + sum) ^ ((v0 >> 5)+k3);
  v0 +=((v1 << 4)+k0) ^ (v1 + sum) ^ ((v1 >> 5)+k1);
  v1 +=((v0 << 4)+k2) ^ (v0 + sum) ^ ((v0 >> 5)+k3);
  v0 +=((v1 << 4)+k0) ^ (v1 + sum) ^ ((v1 >> 5)+k1);
  v1 +=((v0 << 4)+k2) ^ (v0 + sum) ^ ((v0 >> 5)+k3);
  v0 +=((v1 << 4)+k0) ^ (v1 + sum) ^ ((v1 >> 5)+k1);
  v1 +=((v0 << 4)+k2) ^ (v0 + sum) ^ ((v0 >> 5)+k3);
  return ivec2 (v0, v1);
}

vec2 rand2(float entropy, float entropy2) {
  uint v0 = seed0; // + floatBitsToInt(entropy);
  uint v1 = seq++; // + floatBitsToInt(entropy2);
  v0 +=((v1 << 4)+k0) ^ (v1 + sum) ^ ((v1 >> 5)+k1);
  v1 +=((v0 << 4)+k2) ^ (v0 + sum) ^ ((v0 >> 5)+k3);
  v0 +=((v1 << 4)+k0) ^ (v1 + sum) ^ ((v1 >> 5)+k1);
  v1 +=((v0 << 4)+k2) ^ (v0 + sum) ^ ((v0 >> 5)+k3);
  v0 +=((v1 << 4)+k0) ^ (v1 + sum) ^ ((v1 >> 5)+k1);
  v1 +=((v0 << 4)+k2) ^ (v0 + sum) ^ ((v0 >> 5)+k3);
  v0 +=((v1 << 4)+k0) ^ (v1 + sum) ^ ((v1 >> 5)+k1);
  v1 +=((v0 << 4)+k2) ^ (v0 + sum) ^ ((v0 >> 5)+k3);
  float fv0 = float(v0 >> 2);
  float fv1 = float(v1 >> 2);
  return vec2 (fv0 / 1073741824.f, fv1 / 1073741824.f);
}

float rand(float entropy) {
  return rand2(entropy, entropy).y;
}

float srand(float entropy) {
  return rand(entropy) * 2.f - 1.f;
}


vec2 normal_rand(float entropy, float entropy2) {
  vec2 rr = rand2(entropy, entropy2);
  if (rr.x == 0) return vec2(0,0);
  float r = sqrt(-2 * log(rr.x));
  float a = rr.y * 2 * PI;

  return vec2 (cos(a) * r, sin(a) * r);
}

vec3 wall_distr(in vec3 pos) {
  vec2 n1 = normal_rand(pos.z, pos.x);
  vec2 n2 = normal_rand(pos.y, pos.y);
  return vec3 (
      n1.x * wall_distribution,
      n2.x * wall_distribution,
      n2.y * wall_distribution);
}

vec3 light_distr(in vec3 point) {
  return vec3(
      srand(point.z) * light_size,
      srand(point.x) * light_size,
      srand(point.y) * light_size);
}

float lense_gen_r(in float a) {
  return sqrt(rand(a)) * lense_blur;
}

float lense_gen_a(in float a) {
  return rand(a) * 2 * PI;
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
std::uniform_real_distribution<float> lense_gen_r{0,1};
std::uniform_real_distribution<float> lense_gen_a{0,2 * M_PI};
std::uniform_real_distribution<float> reflect_gen{0.f, 1.f};

std::normal_distribution<float> wall_gen{0, 1};
std::uniform_real_distribution<float> light_gen{-light_size, light_size};
std::uniform_real_distribution<float> antialiasing{-0.5,0.5};

vec3 wall_distr() {
  auto d = []() {
    return wall_gen(gen);
  };

  return vec3(
      d() * wall_distribution,
      d() * wall_distribution,
      d() * wall_distribution);
}

vec3 light_distr() {
    return vec3(light_gen(gen), light_gen(gen), light_gen(gen));
}

#endif  // USE_HW

#endif
