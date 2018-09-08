#include "common.h"

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

SineHit sine_hit(in vec3 norm_ray, in vec3 origin) {
  float h = 0.4;
  float dh = 0.4;
  float ball_size2 = dh * dh;
  float max_z = h + dh;
  float min_z = h - dh;
  float period = dh * 2;
  bool going_in = true;

  float dist0 = (max_z - origin.z) / norm_ray.z;
  float dist1 = (min_z - origin.z) / norm_ray.z;
  dist0 = std::max(0.f, dist0);
  dist1 = std::max(0.f, dist1);
  if (dist0 == 0 && dist1 == 0) {
    return SineHit(max_distance, vec3(), vec3());
  }
  float x0 = origin.x + norm_ray.x * dist0;
  float x1 = origin.x + norm_ray.x * dist1;
  if (norm_ray.z < 0) {
    assert(dist0 <= dist1);
    if (x0 > x1) std::swap(x0, x1);
    // FIXME: compress x dimension
    for (int i = std::floor(x0/period); i < std::ceil(x1/period); i++) {
      if (i != 3 && i != 4) continue;
      float center_x = (i + 0.5) * period;
      float center_z = h;
      float dx = center_x - origin.x;
      float dz = center_z - origin.z;
      float closest_point_distance_from_viewer = dx * norm_ray.x + dz * norm_ray.z;
      if (closest_point_distance_from_viewer < 0) {
        continue;
      }
      float ray_projection = sqrt(norm_ray.x * norm_ray.x + norm_ray.z * norm_ray.z);
      closest_point_distance_from_viewer /= ray_projection;
      float ball_distance2 = dx * dx + dz * dz;
      float distance_from_object_center2 = ball_distance2 -
        closest_point_distance_from_viewer * closest_point_distance_from_viewer;
      if (distance_from_object_center2 > ball_size2) {
        continue;
      }
      float distance_from_origin = closest_point_distance_from_viewer -
        sqrt(ball_size2 - distance_from_object_center2);
      distance_from_origin /= ray_projection;
      vec3 pos = origin + norm_ray * distance_from_origin;
      if (pos.y < room.a_.y || pos.y > room.b_.y) {
        return SineHit(max_distance, vec3(), vec3());
      }

      return SineHit(-1, pos, vec3(center_x, pos.y, center_z));
    }
  } else {
    return SineHit(max_distance, vec3(), vec3());
  }

  return SineHit(max_distance, vec3(), vec3());
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

  return light_color * (1 / (total_distance * total_distance));
}

#include "stages.h"
