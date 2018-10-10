#include "optix/shared.h"

#define LENGTH(a) (sizeof(a)/sizeof(a[0]))
#define MAX_FLOAT 1e37
#define sign(x) ((x) > 0 ? 1.f : -1.f)
#define inversesqrt(x) (1.f / sqrtf(x));
#define REF(x) x&
#define swap(a,b) { float x = a; a = b; b = x; }

using namespace optix;
using namespace std;

rtBuffer<float4, 2> sysOutputBuffer; // RGBA32F
rtBuffer<float4, 2> sysAlbedoBuffer; // RGBA32F
rtBuffer<float4, 2> sysOutNormalBuffer;

#define TEXTURE(wall)                                   \
rtTextureSampler<float4, 2> wall##_albedo;              \
rtTextureSampler<float4, 2> wall##_normals;             \
rtTextureSampler<float,  2> wall##_roughness;           \
rtDeclareVariable(float, wall##_specular_exponent, , ); \
rtDeclareVariable(float, wall##_diffuse_ammount, , );

TEXTURE(wall)
TEXTURE(ceiling)
TEXTURE(floor)

rtDeclareVariable(uint2, theLaunchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, theLaunchDim,   rtLaunchDim, );

rtDeclareVariable(uint, sysMaxRays, , );
rtDeclareVariable(uint, sysMaxDepth, , );

rtDeclareVariable(float3, sysViewer, , );
rtDeclareVariable(float3, sysSight, , );
rtDeclareVariable(float3, sysSightX, , );
rtDeclareVariable(float3, sysSightY, , );
rtDeclareVariable(float,  sysLightSize, , );
rtDeclareVariable(float,  sysLightSize2, ,);
rtDeclareVariable(float,  sysLenseBlur, , );
rtDeclareVariable(float,  sysFocusedDistance, , );
rtDeclareVariable(uint,   sysBatchSize, , );
rtDeclareVariable(uint,   sysFrameNum, , );
rtDeclareVariable(uint,   sysMaxInternalReflections, , );
rtDeclareVariable(float,  sysRefractionIndex, , );
rtDeclareVariable(float3, sysAbsorption, , make_float3(0.17,0.17,0.53));
rtDeclareVariable(uint,   sysTracerFlags, , );
rtDeclareVariable(float, sysTime, , );

//rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );


struct PerRayData {
};

//rtDeclareVariable(PerRayData, thePrd, rtPayload, );

///  struct_input.h ////
struct Material {
  float diffuse_ammount_;
//  float scattering_;
  float specular_exponent_;
};

struct Box {
  float3 a_;
  float3 b_;
};



struct Ball {
  float3 position_;
  float3 color_;
  float size_;
  float size2_;
  float inv_size_;
  Material material_;
};

struct Balls {
  Ball ball[3];
};

rtDeclareVariable(Balls, balls, ,);
rtDeclareVariable(Box, bbox, ,);
rtDeclareVariable(Box, room, ,);

__device__
const float fov = 0.7;
__device__
const float light_power = 200.4f;
__device__
const float3 light_pos {5.0, -8, 3.0};
__device__
const float3 light_color {light_power, light_power, light_power};

struct Hit {
  int id_;
  float closest_point_distance_from_viewer_;
  float distance_from_object_center2_;
};

struct RoomHit {
  float min_dist;
  float3 intersection;
  float3 normal;
  float3 reflection;
  float3 color;
  Material material;
};

__device__
const float max_distance = 1000;
__device__
const Hit no_hit = Hit{-1, max_distance, 0};


template<int N>
RT_FUNCTION
uint tea(uint v0, uint v1) {
  uint k0 = 0xA341316C;
  uint k1 = 0xC8013EA4;
  uint k2 = 0xAD90777D;
  uint k3 = 0x7E95761E;
  uint sum = 0x9e3779b9;

  for (int i = 0; i < N; i++) {
    v0 +=((v1 << 4)+k0) ^ (v1 + sum) ^ ((v1 >> 5)+k1);
    v1 +=((v0 << 4)+k2) ^ (v0 + sum) ^ ((v0 >> 5)+k3);
  }
  return v0;
}


RT_FUNCTION
float rand1(uint& seed) {
  // https://en.wikipedia.org/wiki/Linear_congruential_generator
  uint LCG_mul = 1664525;
  uint LCG_incr = 1013904223;
  uint LCG_mask = 0xFFFFFF; // 24 lower bits
  float LCG_normalizer = 5.960464477539063e-08f;//1.f/(float)LCG_mask;

  seed = seed * LCG_mul + LCG_incr;
  return (seed & LCG_mask) * LCG_normalizer;
}

RT_FUNCTION
float2 rand2(uint& seed) {
  return optix::float2{rand1(seed), rand1(seed)};
}

RT_FUNCTION
float srand1(uint& seed) {
  return rand1(seed) * 2.f - 1.f;
}


RT_FUNCTION
float2 normal_rand(uint& seed) {
  float2 rr = rand2(seed);
  if (rr.x == 0) return float2{0,0};
  float r = sqrtf(-2 * log(rr.x));
  float a = rr.y * 2 * M_PI;

  return float2{cosf(a) * r, sinf(a) * r};
}

RT_FUNCTION
float3 wall_distr(uint& seed, float scattering) {
  float2 n1 = normal_rand(seed);
  float2 n2 = normal_rand(seed);
  return float3 {
      n1.x * scattering,
      n2.x * scattering,
      n2.y * scattering};
}

RT_FUNCTION
float3 light_distr(uint& seed) {
  while (true) {
    float x = srand1(seed);
    float y = srand1(seed);
    float z = srand1(seed);
    if (x*x + y*y + z*z <= 1) {
      return float3{x,y,z} * sysLightSize;
    }
  }
}

RT_FUNCTION
float lense_gen_r(uint& seed) {
  return sqrtf(rand1(seed)) * sysLenseBlur;
}

RT_FUNCTION
float lense_gen_a(uint& seed) {
  return rand1(seed) * 2 * M_PI;
}

RT_FUNCTION
float antialiasing(uint& seed) {
  return srand1(seed) * 0.5;
}

RT_FUNCTION
float reflect_gen(uint& seed) {
  return rand1(seed);
}

//float OBJECT_REFLECTIVITY = 0;

// https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel
RT_FUNCTION
float3 refract(float ior, float3 N, float3 I) {
  float cosi = clamp(dot(I, N), -1.f, 1.f);
  float etai = 1, etat = ior;
  float3 n = N;
  if (cosi < 0) { cosi = -cosi; } else { swap(etai, etat); n= -N; }
  float eta = etai / etat;
  float k = 1 - eta * eta * (1 - cosi * cosi);
//  assert(k >= 0);
  return normalize(I * eta + n * (eta * cosi - sqrtf(k)));
}

RT_FUNCTION
float fresnel(float ior, float3 N, float3 I) {
  float cosi = clamp(dot(I, N), -1.f, 1.f);
  float etai = 1, etat = ior;
  float kr;
  if (cosi > 0) { swap(etai, etat); }
  // Compute sini using Snell's law
  float sint = etai / etat * sqrt(max(0.f, 1 - cosi * cosi));
  // Total internal reflection
  if (sint >= 1) {
    kr = 1;
  } else {
    float cost = sqrt(max(0.f, 1 - sint * sint));
    cosi = abs(cosi);
    float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
    float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
    kr = (Rs * Rs + Rp * Rp) / 2;
  }
  // As a consequence of the conservation of energy, transmittance is given by:
  // kt = 1 - kr;
  return kr;
}

RT_FUNCTION
Hit ball_hit(const int id, const float3 norm_ray, const float3 origin) {
  float3 ball_vector = balls.ball[id].position_ - origin;

  float closest_point_distance_from_viewer = dot(norm_ray, ball_vector);
  if (closest_point_distance_from_viewer < 0) {
    return no_hit;
  }

  float ball_distance2 = dot(ball_vector, ball_vector);
  float distance_from_object_center2 = ball_distance2 -
    closest_point_distance_from_viewer * closest_point_distance_from_viewer;
  if (distance_from_object_center2 > balls.ball[id].size2_) {
    return no_hit;
  }
  return Hit{id, closest_point_distance_from_viewer, distance_from_object_center2};
}

RT_FUNCTION
Hit bbox_hit(const float3 norm_ray, const float3 origin) {
  float3 tMin = (bbox.a_ - origin) / norm_ray;
  float3 tMax = (bbox.b_ - origin) / norm_ray;
  float3 t1 = make_float3(
      min(tMin.x, tMax.x),
      min(tMin.y, tMax.y),
      min(tMin.z, tMax.z));
  float3 t2 = make_float3(
      max(tMin.x, tMax.x),
      max(tMin.y, tMax.y),
      max(tMin.z, tMax.z));
  float tNear = max(max(t1.x, t1.y), t1.z);
  float tFar = min(min(t2.x, t2.y), t2.z);
  if (tFar < 0 || tNear > tFar) return no_hit;

  Hit hit = no_hit;
  for (size_t i = 0; i < LENGTH(balls.ball); i++) {
    Hit other_hit = ball_hit(i, norm_ray, origin);
    if (other_hit.closest_point_distance_from_viewer_ <
        hit.closest_point_distance_from_viewer_) {
      hit = other_hit;
    }
  }
  return hit;
}

RT_FUNCTION
RoomHit room_hit_internal(
    const float3 intersection,
    const float3 norm_ray,
    const float3 normal,
    const float3 U,
    const float3 V,
    rtTextureSampler<float4, 2> albedo,
    rtTextureSampler<float4, 2> normals,
    rtTextureSampler<float, 2> roughnesses,
    float u,
    float v,
    float min_dist,
    float specular_exponent,
    float diffuse_ammount) {
  float3 color = make_float3(tex2D(albedo, u, v));
  float4 nn = tex2D(normals, u, v);
  float roughness = tex2D(roughnesses, u, v);
  float3 n = normalize(normal * nn.z + U * nn.x + V * nn.y);
//  n = normal;

  Material material;
//  // FIXME: change texture mode instead of *256
  material.specular_exponent_ = 1 + specular_exponent * (1 - roughness);
  material.diffuse_ammount_ = diffuse_ammount;
//  Material material;
//  material.specular_exponent_ = 32;
//  material.diffuse_ammount_ = 0.3;

  float3 reflection = norm_ray - n * (dot(norm_ray, n) * 2);
//  assert(isfinite(min_dist));
  return RoomHit{min_dist, intersection, n, reflection, color, material};
}

RT_FUNCTION
RoomHit room_hit(const float3 norm_ray, const float3 origin) {
  float3 tMin = (room.a_ - origin) / norm_ray;
  float3 tMax = (room.b_ - origin) / norm_ray;
//  float3 t1 = min(tMin, tMax);
  float3 t2 = make_float3(
      max(tMin.x, tMax.x),
      max(tMin.y, tMax.y),
      max(tMin.z, tMax.z));
  if (!isfinite(t2.x)) t2.x = MAX_FLOAT;
  if (!isfinite(t2.y)) t2.y = MAX_FLOAT;
  if (!isfinite(t2.z)) t2.z = MAX_FLOAT;
//  float tNear = max(max(t1.x, t1.y), t1.z);
//  float tFar = min(min(t2.x, t2.y), t2.z);

  float min_dist;
  float3 normal;
  float3 U, V;

#define FINISH(tex, u_expr, v_expr) {                   \
  float3 intersection = origin + norm_ray * min_dist;   \
  float u = (u_expr);                                   \
  float v = (v_expr);                                   \
                                                        \
  return room_hit_internal(                             \
      intersection,                                     \
      norm_ray,                                         \
      normal,                                           \
      U, V,                                             \
      tex##_albedo, tex##_normals,tex##_roughness,      \
      u, v, min_dist, tex##_specular_exponent, tex##_diffuse_ammount); \
}

  if (t2.y < t2.z) {
    if (t2.x < t2.y) {
      min_dist = t2.x;
      normal = float3{sign(-norm_ray.x), 0, 0};
      U = float3{0, 1, 0};
      V = float3{0, 0, 1};
      FINISH(wall, (intersection.x + intersection.y) * 0.5f, intersection.z * 0.5f);
    } else {
      min_dist = t2.y;
      normal = float3{0, sign(-norm_ray.y), 0};
      U = float3{1, 0, 0};
      V = float3{0, 0, 1};
      FINISH(wall, (intersection.x + intersection.y) * 0.5f, intersection.z * 0.5f);
    }
  } else {
    if (t2.x < t2.z) {
      min_dist = t2.x;
      normal = float3{sign(-norm_ray.x), 0, 0};
      U = float3{0, 1, 0};
      V = float3{0, 0, 1};
      FINISH(wall, (intersection.x + intersection.y) * 0.5f, intersection.z * 0.5f);
    } else {
      min_dist = t2.z;
      normal = float3{0, 0, sign(-norm_ray.z)};
      if (norm_ray.z > 0) {
        U = float3{1, 0, 0};
        V = float3{0, 1, 0};
        FINISH(ceiling, intersection.x * 0.2f, intersection.y * 0.2f);
      } else {
        float3 intersection = origin + norm_ray * min_dist;
        float3 n{0.f,0.f,1.f};
        for (int dx = -1; dx < 2; dx++) {
          for (int dy = -1; dy < 2; dy++) {
            int2 pos {int(floor(intersection.x)) + dx, int(floor(intersection.y)) + dy};
            uint seed = tea<1>(pos.x, pos.y);
            float duration = 0.9 + rand1(seed) * 0.1;
            float start = floor(sysTime / duration) * duration;
            seed = tea<1>(pos.x, pos.y + uint(start * 3));
            float radius = sysTime - start;
            float3 fpos = make_float3(make_float2(pos) + rand2(seed), 0);
            float3 vec = intersection - fpos;
            float dist = sqrt(dot(vec, vec));
            float dist_from_radius = abs(dist - radius);
            float intensity = max(0.f, 0.1f - 0.5 * dist_from_radius) * (0.9 - radius);

            float3 dir = vec / dist;
            n+= dir * (sin(dist_from_radius * 50) * intensity);
          }
        }
        n = normalize(n);
        float3 reflection = norm_ray - n * (dot(norm_ray, n) * 2);
        float3 color = make_float3(1);
        Material m;
        m.diffuse_ammount_ = 0;
        m.specular_exponent_ = 100000;
        return RoomHit{min_dist, intersection, n, reflection, color, m};
      }
    }
  }
}

RT_FUNCTION
Hit light_hit(const float3 norm_ray, const float3 origin) {
  float3 light_vector = light_pos - origin;
  float light_distance2 = dot(light_vector, light_vector);

  float closest_point_distance_from_origin = dot(norm_ray, light_vector);
  if (closest_point_distance_from_origin < 0) {
    return no_hit;
  }

  float distance_from_light_center2 = light_distance2 -
    closest_point_distance_from_origin * closest_point_distance_from_origin;
  if (distance_from_light_center2 > sysLightSize2) {
    return no_hit;
  }
  return Hit{-2, closest_point_distance_from_origin, distance_from_light_center2};
}

RT_FUNCTION
float3 scatter(uint& seed, const float3 v, float specular_exponent) {
  // https://en.wikipedia.org/wiki/Specular_highlight#Phong_distribution
  float N = specular_exponent;
  float r = reflect_gen(seed);
  // https://programming.guide/random-point-within-circle.html
  float cos_b = specular_exponent != 0 ? powf(r, 1/(N+1)) : r * 2 - 1;
  float sin_b = sqrt(1 - cos_b * cos_b);
  // https://scicomp.stackexchange.com/questions/27965/rotate-a-vector-by-a-randomly-oriented-angle
  float a = reflect_gen(seed) * 2 * M_PI;
  float sin_a = sin(a);
  float cos_a = cos(a);
  float3 rv = make_float3(0,0,1);
  if (v.y < v.z) {
    if (v.x < v.y) {
      rv = make_float3(1,0,0);
    } else {
      rv = make_float3(0,1,0);
    }
  }
  float3 xv = normalize(cross(v, rv));
  float3 yv = cross(v, xv);
  float3 res =  xv * (sin_b * sin_a) + yv * (sin_b *cos_a) + v * cos_b;
  return res;
}

#define FLAG_TERMINATE 1
#define FLAG_NO_SECONDARY 2
#define FLAG_ALBEDO 4

struct RayData {
  float3 intensity;
  float3 albedo;
  float3 result_normal;

  float3 origin;
  float3 norm_ray;
  // FIXME:add norm_ray_inv as well
  float3 color_filter;
  float distance_from_eye;
  int flags;
  uint seed;
};

RT_FUNCTION
void compute_light(
    REF(RayData) ray,
    const float3 color,
    const float3 specular_color,
    const Material m,
    const float3 normal) {
  if ((ray.flags & FLAG_ALBEDO) == 0) {
    ray.flags |= FLAG_ALBEDO;
    ray.albedo = color;
    ray.result_normal = normal;
  }
  if ((ray.flags & FLAG_NO_SECONDARY) == 0) {
    float3 light_rnd_pos = light_pos + light_distr(ray.seed);

    float3 light_from_point = light_rnd_pos - ray.origin;
    float angle_x_distance = dot(normal, light_from_point);
    if (angle_x_distance > 0) {

      float light_distance2 = dot(light_from_point, light_from_point);
      float light_distance_inv = inversesqrt(light_distance2);
      float light_distance = 1.f/light_distance_inv;
      float3 light_from_point_norm = light_from_point * light_distance_inv;

      Hit hit = bbox_hit(light_from_point_norm, ray.origin);

      if (hit.closest_point_distance_from_viewer_ > light_distance) {
        float angle = angle_x_distance * light_distance_inv;
        float a = dot(ray.norm_ray, light_from_point_norm);
        float specular = 0;
        if (a > 0) {
          // Clamp
          a = min(a, 1.f);
          specular = powf(a, m.specular_exponent_);
        }
        float total_distance = light_distance + ray.distance_from_eye;

        // FIXME: should angle be only used for diffuse color?
        float3 reflected_light = (color * light_color) * m.diffuse_ammount_
          + (specular_color * light_color) * (1-m.diffuse_ammount_) * specular;
        ray.intensity += reflected_light * ray.color_filter * (angle / (total_distance * total_distance + 1e-12f));
      }
    }
  }
  if ((ray.flags & FLAG_TERMINATE) != 0) return;

  float r = reflect_gen(ray.seed);
  bool is_diffuse = r < m.diffuse_ammount_;
    ray.norm_ray = is_diffuse ? scatter(ray.seed, normal, 1) : scatter(ray.seed, ray.norm_ray, m.specular_exponent_);
    float angle = dot(ray.norm_ray, normal);
    if (angle <= 0) {
      ray.flags |= FLAG_TERMINATE;
      return;
    }
    ray.color_filter = ray.color_filter * angle * (is_diffuse ? color : specular_color);
}

RT_FUNCTION
void light_trace_new(
    const Hit p,
    REF(RayData) ray) {
  float distance_from_origin = p.closest_point_distance_from_viewer_ -
    sqrt(sysLightSize2 - p.distance_from_object_center2_);

  float total_distance = ray.distance_from_eye + distance_from_origin;

  ray.intensity += light_color * ray.color_filter * (1.f / (total_distance * total_distance + 1e-12f));
  if ((ray.flags & FLAG_ALBEDO) == 0) {
    ray.flags |= FLAG_ALBEDO;
    ray.albedo = normalize(light_color);
    float3 intersection = ray.origin + ray.norm_ray * distance_from_origin;
    float3 distance_from_light_vector = intersection - light_pos;
    float3 normal = distance_from_light_vector / sysLightSize;
    ray.result_normal = normal;
  }
}

RT_FUNCTION
void room_trace(
    REF(RayData) ray) {
  RoomHit p = room_hit(ray.norm_ray, ray.origin);
  ray.origin = p.intersection;
  ray.norm_ray = p.reflection;
  ray.distance_from_eye += p.min_dist;

  compute_light(
      ray,
      p.color,
      p.color,
      p.material,
      p.normal);
}

RT_FUNCTION
void trace_ball0_internal(REF(RayData) ray, float size) {
  float start_distance = ray.distance_from_eye;
  for (int i = 0; i < sysMaxInternalReflections; i++) {
    float3 ball_vector = balls.ball[0].position_ - ray.origin;
    float closest_point_distance_from_viewer = dot(ray.norm_ray, ball_vector);
    float distance_from_origin = 2 * closest_point_distance_from_viewer;
    float3 intersection = ray.origin + ray.norm_ray * distance_from_origin;
    float3 distance_from_ball_vector = intersection - balls.ball[0].position_;
    float3 normal = normalize(distance_from_ball_vector);
    ray.origin = balls.ball[0].position_ + normal * size;
    ray.distance_from_eye += distance_from_origin;

    if (reflect_gen(ray.seed) < fresnel(sysRefractionIndex, normal, ray.norm_ray)) {
      float3 ray_reflection = ray.norm_ray - normal * (2 * dot(ray.norm_ray, normal));
      // Restart from new point
      ray.norm_ray = ray_reflection;
      continue;
    } else {
      ray.norm_ray = refract(sysRefractionIndex, normal, ray.norm_ray);
      float td = ray.distance_from_eye - start_distance; // travel distance
      float3 extinction = make_float3(
          expf(-sysAbsorption.x * td),
          expf(-sysAbsorption.y * td),
          expf(-sysAbsorption.z * td));
      ray.color_filter = ray.color_filter * extinction;
      return;
    }
  }
  ray.flags |= FLAG_TERMINATE;
}

RT_FUNCTION
void make_refraction(
    REF(RayData) ray,
    float3 normal,
    float size) {
  ray.norm_ray = refract(sysRefractionIndex, normal, ray.norm_ray);
  trace_ball0_internal(ray, size);
}

RT_FUNCTION
void make_reflection(
    REF(RayData) ray,
    const float3 color,
    const Material m,
    const float3 normal) {
  ray.norm_ray = ray.norm_ray - normal * (2 * dot(ray.norm_ray, normal));
  compute_light(ray, color, color, m, normal);
}

RT_FUNCTION
void ball_trace (
    REF(RayData) ray,
    const Hit p) {
  Ball ball = balls.ball[p.id_];
  float distance_from_origin = p.closest_point_distance_from_viewer_ -
    sqrt(ball.size2_ - p.distance_from_object_center2_);

  ray.origin = ray.origin + ray.norm_ray * distance_from_origin;
  ray.distance_from_eye += distance_from_origin;

  float3 distance_from_ball_vector = ray.origin - ball.position_;
  float3 normal = distance_from_ball_vector * ball.inv_size_;

  if (p.id_ != 0) {
    make_reflection(ray, ball.color_, ball.material_, normal);
    return;
  }

  float reflect_ammount = fresnel(sysRefractionIndex, normal, ray.norm_ray);

  if (reflect_gen(ray.seed) < reflect_ammount) {
    make_reflection(ray, ball.color_, ball.material_, normal);
  } else {
    if ((ray.flags & FLAG_ALBEDO) == 0) {
      ray.flags |= FLAG_ALBEDO;
      ray.albedo = ball.color_;
      ray.result_normal = normal;
    }
    make_refraction(ray, normal, ball.size_);
  }
}


RT_FUNCTION
void trace (RayData& ray) {

  int depth = 0;
  while (true) {
    Hit hit = bbox_hit(ray.norm_ray, ray.origin);

    Hit light = light_hit(ray.norm_ray, ray.origin);
    if (light.closest_point_distance_from_viewer_ <
        hit.closest_point_distance_from_viewer_) {
      light_trace_new(light, ray);
      return;
    }

    if (depth == sysMaxDepth - 1) ray.flags |= FLAG_TERMINATE;

    if (hit.id_ < 0) {
      room_trace(ray);
    } else {
      ball_trace(ray, hit);
    }
    if ((ray.flags & FLAG_TERMINATE) != 0) return;
    depth++;
  }
}

RT_PROGRAM void ray() {
  uint2 pixel_coords = theLaunchIndex;
  uint2 dims = theLaunchDim;
  pixel_coords.x *= sysBatchSize;
  dims.x *= sysBatchSize;


  float x = (float((int)pixel_coords.x * 2 - (int)dims.x) / dims.x);
  float y = (float((int)pixel_coords.y * 2 - (int)dims.y) / dims.y);


  float3 xoffset = sysSightX * fov;
  float3 yoffset = -sysSightY * (fov * dims.y / dims.x);
  float3 dx = xoffset * (2.f/dims.x);
  float3 dy = yoffset * (2.f/dims.y);

  float3 ray = sysSight + xoffset * x + yoffset * y ;
  float3 origin = sysViewer;
  uint seed = tea<4>(pixel_coords.y * dims.x + pixel_coords.x, sysFrameNum);

  float weight = sysMaxRays / float(sysFrameNum + sysMaxRays);

  for (int xx = 0; xx < sysBatchSize; xx++) {
    float4 total_intensity = make_float4(0);
    float4 total_albedo = make_float4(0);
    float3 total_normal = make_float3(0);

    for (int i = 0; i < sysMaxRays; i++) {
      // no normalize here to preserve focal plane
      float3 focused_ray = (ray + dx * antialiasing(seed) + dy * antialiasing(seed));
      float3 focused_point = origin + focused_ray * sysFocusedDistance;
      float r = lense_gen_r(seed);
      float a = lense_gen_a(seed) * 1 * M_PI;
      float3 me = origin + sysSightX * (r * cos(a)) + sysSightY * (r * sin(a));
      float3 new_ray = normalize(focused_point - me);

      RayData ray;
      ray.origin = me;
      ray.norm_ray = new_ray;
      ray.distance_from_eye = 0;
      ray.color_filter = float3{1,1,1};
      ray.intensity = float3{0,0,0};
      ray.flags = sysTracerFlags;
      ray.seed = seed;

      trace(ray);
      if (ray.intensity.x < 0 || ray.intensity.y < 0 || ray.intensity.z < 0
          || isnan(ray.intensity.x) || isnan(ray.intensity.y) || isnan(ray.intensity.z)
          || !isfinite(ray.intensity.x) || !isfinite(ray.intensity.z) || !isfinite(ray.intensity.z)) {
        ray.intensity = make_float3(0);
      }
      total_intensity += make_float4(ray.intensity, 1);
      total_albedo += make_float4(ray.albedo, 1);
      total_normal += ray.result_normal;
      seed = ray.seed;
    }
    float scale = 1.f / sysMaxRays;
    sysOutputBuffer[pixel_coords] = lerp(sysOutputBuffer[pixel_coords], total_intensity * scale, weight);
    sysAlbedoBuffer[pixel_coords] = lerp(sysAlbedoBuffer[pixel_coords], total_albedo * scale, weight);

    float3 normal_scaled = total_normal * scale;
    // FIXME: used matrix instead
    float4 normal_screen = make_float4(
        dot(normal_scaled, sysSightX),
        -dot(normal_scaled, sysSightY),
        dot(normal_scaled, sysSight), 0.f);
    sysOutNormalBuffer[pixel_coords] = lerp(sysOutNormalBuffer[pixel_coords], normal_screen, weight);
    pixel_coords.x++;
    ray += dx;
  }
}
