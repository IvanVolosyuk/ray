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
  if (distance_from_object_center2 > ball_size2) {
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
  assert(isfinite(min_dist));
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

#include "stages.h"
