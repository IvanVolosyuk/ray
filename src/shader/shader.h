#include "common.h"

//float OBJECT_REFLECTIVITY = 0;

// https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel
vec3 refract(float ior, vec3 N, vec3 I) { 
  float cosi = clamp(dot(I, N), -1.f, 1.f); 
  float etai = 1, etat = ior; 
  vec3 n = N; 
  if (cosi < 0) { cosi = -cosi; } else { swap(etai, etat); n= -N; } 
  float eta = etai / etat; 
  float k = 1 - eta * eta * (1 - cosi * cosi); 
  assert(k >= 0);
  return normalize(I * eta + n * (eta * cosi - sqrt(k))); 
} 

float fresnel(float ior, vec3 N, vec3 I) { 
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

Hit ball_hit(in int id, in vec3 norm_ray, in vec3 origin) {
  vec3 ball_vector = balls[id].position_ - origin;

  float closest_point_distance_from_viewer = dot(norm_ray, ball_vector);
  if (closest_point_distance_from_viewer < 0) {
    return no_hit;
  }

  float ball_distance2 = dot(ball_vector, ball_vector);
  float distance_from_object_center2 = ball_distance2 -
    closest_point_distance_from_viewer * closest_point_distance_from_viewer;
  if (distance_from_object_center2 > balls[id].size2_) {
    return no_hit;
  }
  return Hit(id, closest_point_distance_from_viewer, distance_from_object_center2);
}

Hit bbox_hit(in vec3 norm_ray, in vec3 origin) {
  vec3 tMin = (bbox.a_ - origin) / norm_ray;
  vec3 tMax = (bbox.b_ - origin) / norm_ray;
  vec3 t1 = min(tMin, tMax);
  vec3 t2 = max(tMin, tMax);
  float tNear = max(max(t1.x, t1.y), t1.z);
  float tFar = min(min(t2.x, t2.y), t2.z);
  if (tFar < 0 || tNear > tFar) return no_hit;

  Hit hit = no_hit;
  for (size_t i = 0; i < LENGTH(balls); i++) {
    Hit other_hit = ball_hit(i, norm_ray, origin);
    if (other_hit.closest_point_distance_from_viewer_ <
        hit.closest_point_distance_from_viewer_) {
      hit = other_hit;
    }
  }
  return hit;
}

RoomHit room_hit_internal(
    in vec3 norm_ray,
    in vec3 origin,
    in float min_dist,
    in vec3 intersection,
    in vec3 normal,
    in vec3 U,
    in vec3 V,
    in uint px,
    in uint nn,
    in float specular_exponent,
    in float diffuse_ammount) {
  vec3 color;
  color.z = float((px >> 16) & 255);
  color.y = float((px >> 8) & 255);
  color.x = float(px & 255);
  color = color * (1.f/256.f);

  float nx = float(nn & 255) - 128;
  float nz = (float((nn>>8) & 255) - 128);
  float ny = float((nn>>16) & 255) - 128;
  vec3 n = normalize(normal * ny + U * nx + V * nz);
  Material material;
  material.specular_exponent_ = 1 + specular_exponent * (256 - float((px >> 24)&255));
  material.diffuse_ammount_ = diffuse_ammount;
  vec3 reflection = norm_ray - n * (dot(norm_ray, n) * 2);
  assert(isfinite(min_dist));
  return RoomHit(min_dist, intersection, n, reflection, color, material);
}

RoomHit room_hit(in vec3 norm_ray, in vec3 origin) {
  vec3 tMin = (room.a_ - origin) / norm_ray;
  vec3 tMax = (room.b_ - origin) / norm_ray;
//  vec3 t1 = min(tMin, tMax);
  vec3 t2 = max(tMin, tMax);
  if (!isfinite(t2.x)) t2.x = MAX_FLOAT;
  if (!isfinite(t2.y)) t2.y = MAX_FLOAT;
  if (!isfinite(t2.z)) t2.z = MAX_FLOAT;
//  float tNear = max(max(t1.x, t1.y), t1.z);
//  float tFar = min(min(t2.x, t2.y), t2.z);

  float min_dist;
  vec3 normal;
  vec3 U, V;

#define FINISH(N, u_expr, v_expr) {                     \
  vec3 intersection = origin + norm_ray * min_dist;     \
  float u = (u_expr);                                   \
  float v = (v_expr);                                   \
  u = u - floor(u);                                     \
  v = v - floor(v);                                     \
  uint dx = uint(width_##N * u);                        \
  uint dy = uint(height_##N * v);                       \
  uint idx = (dx + dy * width_##N) * 2;                 \
  uint px = pixels_##N[idx];                            \
  uint nn = pixels_##N[idx + 1];                        \
  return room_hit_internal(                             \
      norm_ray,                                         \
      origin,                                           \
      min_dist,                                         \
      intersection,                                     \
      normal,                                           \
      U, V,                                             \
      px, nn,                                           \
      specular_exponent_##N,                            \
      diffuse_ammount_##N);                             \
}

  if (t2.y < t2.z) {
    if (t2.x < t2.y) {
      min_dist = t2.x;
      normal = vec3(sign(-norm_ray.x), 0, 0);
      U = vec3(0, 1, 0);
      V = vec3(0, 0, 1);
      FINISH(1, (intersection.x + intersection.y) * 0.5f, intersection.z * 0.5f);
    } else {
      min_dist = t2.y;
      normal = vec3(0, sign(-norm_ray.y), 0);
      U = vec3(1, 0, 0);
      V = vec3(0, 0, 1);
      FINISH(1, (intersection.x + intersection.y) * 0.5f, intersection.z * 0.5f);
    }
  } else {
    if (t2.x < t2.z) {
      min_dist = t2.x;
      normal = vec3(sign(-norm_ray.x), 0, 0);
      U = vec3(0, 1, 0);
      V = vec3(0, 0, 1);
      FINISH(1, (intersection.x + intersection.y) * 0.5f, intersection.z * 0.5f);
    } else {
      min_dist = t2.z;
      normal = vec3(0, 0, sign(-norm_ray.z));
      if (norm_ray.z > 0) {
        U = vec3(1, 0, 0);
        V = vec3(0, 1, 0);
        FINISH(2, intersection.x * 0.2f, intersection.y * 0.2f);
      } else {
        U = vec3(1, 0, 0);
        V = vec3(0, 1, 0);
        FINISH(0, intersection.x * 0.2f, intersection.y * 0.2f);
      }
    }
  }
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
    sqrt(light_size2 - p.distance_from_object_center2_);

//  vec3 intersection = origin + norm_ray * distance_from_origin;
//  vec3 distance_from_light_vector = intersection - light_pos;

//  vec3 normal = distance_from_light_vector * light_inv_size;
//  float angle = -dot(norm_ray, normal);
  float total_distance = distance_from_eye + distance_from_origin;

  vec3 res = light_color * (1.f / (total_distance * total_distance + 1e-12f));
  assert(isfinite(res.size2()));
  return res;
}

vec3 scatter(in vec3 v, float specular_exponent) {
  // https://en.wikipedia.org/wiki/Specular_highlight#Phong_distribution
  float N = specular_exponent;
  float r = reflect_gen(SW(gen));
  // https://programming.guide/random-point-within-circle.html
  float cos_b = specular_exponent != 0 ? pow(r, 1/(N+1)) : r * 2 - 1;
  float sin_b = sqrt(1 - cos_b * cos_b);
  // https://scicomp.stackexchange.com/questions/27965/rotate-a-vector-by-a-randomly-oriented-angle
  float a = reflect_gen(SW(gen)) * 2 * M_PI;
  float sin_a = sin(a);
  float cos_a = cos(a);
  vec3 rv = vec3(0,0,1);
  if (v.y < v.z) {
    if (v.x < v.y) {
      rv = vec3(1,0,0);
    } else {
      rv = vec3(0,1,0);
    }
  }
  vec3 xv = normalize(cross(v, rv));
  vec3 yv = cross(v, xv);
  vec3 res =  xv * (sin_b * sin_a) + yv * (sin_b *cos_a) + v * cos_b;
  return res;
}

#define FLAG_TERMINATE 1

struct RayData {
  vec3 intensity;
  vec3 origin;
  vec3 norm_ray;
  // FIXME:add norm_ray_inv as well
  vec3 color_filter;
  float distance_from_eye;
  int flags;
};

void compute_light(
    REF(RayData) ray,
    in vec3 color,
    in vec3 specular_color,
    in Material m,
    in vec3 normal) {
  vec3 light_rnd_pos = light_pos + light_distr();

  vec3 light_from_point = light_rnd_pos - ray.origin;
  float angle_x_distance = dot(normal, light_from_point);
  if (angle_x_distance > 0) {

    float light_distance2 = dot(light_from_point, light_from_point);
    float light_distance_inv = inversesqrt(light_distance2);
    float light_distance = 1.f/light_distance_inv;
    vec3 light_from_point_norm = light_from_point * light_distance_inv;

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
      vec3 reflected_light = (color * light_color) * m.diffuse_ammount_
        + (specular_color * light_color) * (1-m.diffuse_ammount_) * specular;
      ray.intensity += reflected_light * ray.color_filter * (angle / (total_distance * total_distance + 1e-12f));
    }
  }

  if ((ray.flags & FLAG_TERMINATE) != 0) return;

  float r = reflect_gen(SW(gen));
  bool is_diffuse = r < m.diffuse_ammount_;
    ray.norm_ray = scatter(ray.norm_ray, is_diffuse ? 1 : m.specular_exponent_);
    float angle = dot(ray.norm_ray, normal);
    if (angle <= 0) {
      ray.flags |= FLAG_TERMINATE;
      return;
    }
    ray.color_filter = ray.color_filter * angle * (is_diffuse ? color : specular_color);
}

vec3 light_trace_new(
    in Hit p,
    REF(RayData) ray) {
  float distance_from_origin = p.closest_point_distance_from_viewer_ -
    sqrt(light_size2 - p.distance_from_object_center2_);

//  vec3 intersection = origin + norm_ray * distance_from_origin;
//  vec3 distance_from_light_vector = intersection - light_pos;

//  vec3 normal = distance_from_light_vector * light_inv_size;
//  float angle = -dot(norm_ray, normal);
  float total_distance = ray.distance_from_eye + distance_from_origin;

  vec3 res = ray.intensity + light_color * ray.color_filter * (1.f / (total_distance * total_distance + 1e-12f));
  assert(isfinite(res.size2()));
  return res;
}

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

void trace_ball0_internal(REF(RayData) ray, float size) {
  float start_distance = ray.distance_from_eye;
  for (int i = 0; i < max_internal_reflections; i++) {
    vec3 ball_vector = balls[0].position_ - ray.origin;
    float closest_point_distance_from_viewer = dot(ray.norm_ray, ball_vector);
    float distance_from_origin = 2 * closest_point_distance_from_viewer;
    vec3 intersection = ray.origin + ray.norm_ray * distance_from_origin;
    vec3 distance_from_ball_vector = intersection - balls[0].position_;
    vec3 normal = normalize(distance_from_ball_vector);
    ray.origin = balls[0].position_ + normal * size;
    ray.distance_from_eye += distance_from_origin;

    if (reflect_gen(SW(gen)) < fresnel(glass_refraction_index, normal, ray.norm_ray)) {
      vec3 ray_reflection = ray.norm_ray - normal * (2 * dot(ray.norm_ray, normal));
      // Restart from new point
      ray.norm_ray = ray_reflection;
      continue;
    } else {
      ray.norm_ray = refract(glass_refraction_index, normal, ray.norm_ray);
      float td = ray.distance_from_eye - start_distance; // travel distance
      vec3 extinction = vec3(expf(-absorption.x * td), expf(-absorption.y * td), expf(-absorption.z * td));
      ray.color_filter = ray.color_filter * extinction;
      return;
    }
  }
  ray.flags |= FLAG_TERMINATE;
}

void make_refraction(
    REF(RayData) ray,
    vec3 normal,
    float size) {
  ray.norm_ray = refract(glass_refraction_index, normal, ray.norm_ray);
  trace_ball0_internal(ray, size);
}

void make_reflection(
    REF(RayData) ray,
    in vec3 color,
    in Material m,
    in vec3 normal) {
  ray.norm_ray = ray.norm_ray - normal * (2 * dot(ray.norm_ray, normal));
  compute_light(ray, color, color, m, normal);
}

void ball_trace (
    REF(RayData) ray,
    in Hit p) {
  Ball ball = balls[p.id_];
  float distance_from_origin = p.closest_point_distance_from_viewer_ -
    sqrt(ball.size2_ - p.distance_from_object_center2_);

  ray.origin = ray.origin + ray.norm_ray * distance_from_origin;
  ray.distance_from_eye += distance_from_origin;

  vec3 distance_from_ball_vector = ray.origin - ball.position_;
  vec3 normal = distance_from_ball_vector * ball.inv_size_;

  if (p.id_ != 0) {
    make_reflection(ray, ball.color_, ball.material_, normal);
    return;
  }

  float reflect_ammount = fresnel(glass_refraction_index, normal, ray.norm_ray);

  if (reflect_gen(SW(gen)) < reflect_ammount) {
    make_reflection(ray, ball.color_, ball.material_, normal);
  } else {
    make_refraction(ray, normal, ball.size_);
  }
}


vec3 trace (RayData ray) {

  int depth = 0;
  while (true) {
    Hit hit = bbox_hit(ray.norm_ray, ray.origin);

    Hit light = light_hit(ray.norm_ray, ray.origin);
    if (light.closest_point_distance_from_viewer_ <
        hit.closest_point_distance_from_viewer_) {
      return light_trace_new(light, ray);
    }

    if (depth == max_depth - 1) ray.flags |= FLAG_TERMINATE;

    if (hit.id_ < 0) {
      room_trace(ray);
    } else {
      ball_trace(ray, hit);
    }
    if ((ray.flags & FLAG_TERMINATE) != 0) return ray.intensity;
    depth++;
  }
  return ray.intensity;
}

vec3 trace_new (
    in vec3 norm_ray,
    in vec3 origin) {
  RayData ray;
  ray.origin = origin;
  ray.norm_ray = norm_ray;
  ray.distance_from_eye = 0;
  ray.color_filter = vec3(1,1,1);
  ray.intensity = vec3(0,0,0);
  ray.flags = 0;

  vec3 color = trace(ray);
  return max(color, vec3(0));
}
