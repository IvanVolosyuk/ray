#include "common.h"

#define RECURSIVE(name, stage) name ## _ ## stage

#ifndef COMPILE_RECURSION
// Just to make syntax checker happy when working with the file
#define CURR(a) a ## _CURR
#define NEXT(a) a ## _NEXT
vec3 trace_NEXT(in vec3 norm_ray, in vec3 origin, float distance_from_eye);
vec3 trace_all_CURR(in vec3 norm_ray, in vec3 origin, float distance_from_eye);
Hit bbox_hit(in vec3 norm_ray, in vec3 origin);
#endif

vec3 CURR(compute_light) (
    in vec3 color,
    in vec3 specular_color,
    in Material m,
    in vec3 normal,
    in vec3 norm_ray,
    in vec3 reflection,
    in vec3 point,
    bool rought_surface,
    in float distance_from_eye) {
  vec3 total_color = black;

#ifdef NEXT
  float r = reflect_gen(HW(origin)SW(gen));
  if (r < m.diffuse_ammount_) {
    vec3 second_ray_dir = scatter(reflection, 0);
    float angle = dot(second_ray_dir, normal);
    if (angle > 0) {
      assert(isfinite(second_ray_dir.size()));
      vec3 second_ray_color = NEXT(trace)(second_ray_dir, point, distance_from_eye);
      total_color = (color * second_ray_color) * angle;
    }
  } else {
    vec3 second_ray_dir = scatter(reflection, m.specular_exponent_);
    float angle = dot(second_ray_dir, normal);
    if (angle > 0) {
      assert(isfinite(second_ray_dir.size()));
      vec3 second_ray_color = NEXT(trace)(second_ray_dir, point, distance_from_eye);
      total_color = (second_ray_color * specular_color) * angle;
    }
  }
#endif

  vec3 light_rnd_pos = light_pos + light_distr();
  assert(isfinite(light_rnd_pos.size2()));
  assert(isfinite(point.size2()));

  vec3 light_from_point = light_rnd_pos - point;
  assert(isfinite(light_from_point.size2()));
  float angle_x_distance = dot(normal, light_from_point);
  assert(isfinite(angle_x_distance));
  if (angle_x_distance < 0) {
    assert(isfinite(total_color.size2()));
    return total_color;
  }

  float light_distance2 = dot(light_from_point, light_from_point);
  float light_distance_inv = inversesqrt(light_distance2);
  assert(isfinite(light_distance_inv));
  float light_distance = 1.f/light_distance_inv;
  vec3 light_from_point_norm = light_from_point * light_distance_inv;

  Hit hit = bbox_hit(light_from_point_norm, point);
  if (hit.closest_point_distance_from_viewer_ < light_distance) {
    assert(isfinite(total_color.size2()));
    return total_color;
  }

  float angle = angle_x_distance * light_distance_inv;
  float a = dot(reflection, light_from_point_norm);
  float specular = 0;
  if (a > 0) {
    // Clamp
    a = std::min(a, 1.f);

    assert(isfinite(a));
    assert(isfinite(m.specular_exponent_));
    specular = pow(a, m.specular_exponent_);
  }
  assert(isfinite(specular));
  assert(isfinite(angle));
  assert(isfinite(color.size2()));
  assert(isfinite(light_color.size2()));
  float total_distance = light_distance + distance_from_eye;
  assert(isfinite(total_distance));
//  vec3 diffuse_color = (color * light_color) *
//    (angle * specular / (total_distance * total_distance + 1e-12f) * m.diffuse_attenuation_);
  
  // FIXME: should angle be only used for diffuse color?
  vec3 reflected_light = (color * light_color) * m.diffuse_ammount_ + light_color * (1-m.diffuse_ammount_) * specular;
  vec3 diffuse_color = reflected_light * (angle / (total_distance * total_distance + 1e-12f));
  assert(diffuse_color.x >= 0);
  assert(diffuse_color.y >= 0);
  assert(diffuse_color.z >= 0);
  assert(isfinite(total_color.size2()));
  total_color += diffuse_color;
  assert(isfinite(total_color.size2()));
  return total_color;
}

vec3 CURR(trace_ball0_internal)(
    HW(in) vec3 norm_ray,
    HW(in) vec3 origin,
    HW(in) float distance_from_eye) {
#ifdef NEXT
  for (int i = 0; i < max_internal_reflections; i++) {
    vec3 ball_vector = balls[0].position_ - origin;
    float closest_point_distance_from_viewer = dot(norm_ray, ball_vector);
    float distance_from_origin = 2 * closest_point_distance_from_viewer;
    vec3 intersection = origin + norm_ray * distance_from_origin;
    vec3 distance_from_ball_vector = intersection - balls[0].position_;
    vec3 normal = normalize(distance_from_ball_vector);
    intersection = balls[0].position_ + normal * ball_size;

    if (reflect_gen(SW(gen)) < fresnel(glass_refraction_index, normal, norm_ray)) {
      vec3 ray_reflection = norm_ray - normal * (2 * dot(norm_ray, normal));
      assert(ray_reflection.size2() > 0.99 && ray_reflection.size2() < 1.01);
      // Restart from new point
      norm_ray = ray_reflection;
      origin = intersection;
//      origin = balls[0].position_ + normalize(origin - balls[0].position_) * ball_size;
//      assert((origin - balls[0].position_).size2() > 0.99*ball_size2 && (origin - balls[0].position_).size2() < 1.01*ball_size2);
      distance_from_eye += distance_from_origin;
      continue;
    } else {
      vec3 refracted_ray_norm = refract(glass_refraction_index, normal, norm_ray);
      return NEXT(trace)(refracted_ray_norm, intersection, distance_from_eye + distance_from_origin);
    }
  }
#endif
  return vec3(0);
}

vec3 CURR(make_reflection)(
    in vec3 color,
    in Material m,
    in vec3 norm_ray,
    in vec3 normal,
    in vec3 intersection,
    in float distance_from_eye) {
  // FIXME: use reflect
  vec3 ray_reflection = norm_ray - normal * (2 * dot(norm_ray, normal));
  return CURR(compute_light)(
      color,
      color,
      m,
      normal,
      norm_ray,
      ray_reflection,
      intersection,
      false,
      distance_from_eye);
}

vec3 CURR(make_refraction)(
    vec3 norm_ray,
    vec3 normal,
    vec3 intersection,
    float distance_from_eye) {
  vec3 refracted_ray_norm = refract(glass_refraction_index, normal, norm_ray);
  return CURR(trace_ball0_internal)(
      refracted_ray_norm,
      intersection,
      distance_from_eye);
}

vec3 CURR(ball_trace) (
    in Hit p,
    in vec3 norm_ray,
    in vec3 origin,
    float distance_from_eye) {
  Ball ball = balls[p.id_];
  float distance_from_origin = p.closest_point_distance_from_viewer_ -
    sqrt(ball_size2 - p.distance_from_object_center2_);

  vec3 intersection = origin + norm_ray * distance_from_origin;
  vec3 distance_from_ball_vector = intersection - ball.position_;
  vec3 normal = distance_from_ball_vector * ball_inv_size;
  float total_distance = distance_from_eye + distance_from_origin;

  if (p.id_ != 0) {
    return CURR(make_reflection)(ball.color_, ball.material_, norm_ray, normal, intersection, total_distance);
  }

  float reflect_ammount = fresnel(glass_refraction_index, normal, norm_ray);


#ifdef MAX_STAGE
  // Trace both if first ray
  return CURR(make_reflection)(
      ball.color_,
      ball.material_,
      norm_ray, normal,
      intersection,
      total_distance) * reflect_ammount +
    CURR(make_refraction)(
        norm_ray,
        normal,
        intersection,
        total_distance) * (1 - reflect_ammount);
#else
  if (reflect_gen(SW(gen)) < reflect_ammount) {
    return CURR(make_reflection)(ball.color_, ball.material_, norm_ray, normal, intersection, total_distance) * 0.9f;
  } else {
    return CURR(make_refraction)(norm_ray, normal, intersection, total_distance) * 0.9f;
  }
#endif
}

vec3 CURR(room_trace) (
    in vec3 norm_ray,
    in vec3 origin,
    float distance_from_eye) {
  RoomHit p = room_hit(norm_ray, origin);
  assert(isfinite(norm_ray.size()));
  assert(isfinite(p.min_dist));
  vec3 ray = norm_ray * p.min_dist;
  vec3 intersection = origin + ray;
  // tiles
  vec3 color = p.color;
  Material material = room_material;
  vec3 normal = p.normal;
  vec3 reflection = p.reflection;
  std::tuple<vec3,vec3,float,float> tex_lookup;

  // Hack for bumpmap for floor
  if (normal.z == 0) {
    tex_lookup = wall_tex->Get((intersection.x + intersection.y)/2, intersection.z/2, normal);
  } else if (normal.z == -1) {
    tex_lookup = ceiling_tex->Get(intersection.x / 5, intersection.y / 5, normal);
  } else {
    tex_lookup = floor_tex->Get(intersection.x /5, intersection.y / 5, normal);
  }
  color = std::get<0>(tex_lookup);
  assert(color.size2() < 10.05);
  normal = std::get<1>(tex_lookup);
  material.specular_exponent_ = std::get<2>(tex_lookup);
  material.diffuse_ammount_ = std::get<3>(tex_lookup);
  reflection = norm_ray - normal * (dot(norm_ray, normal) * 2);

  return CURR(compute_light)(
      color,
      vec3(1,1,1),
      material,
      normal,
      norm_ray,
      reflection,
      intersection,
      true,
      p.min_dist);
}

vec3 CURR(sine_trace)(
    in SineHit hit,
    vec3 norm_ray,
    vec3 origin,
    float distance_from_eye) {
  vec3 normal = normalize(hit.point - hit.center);
  vec3 ray_reflection = norm_ray - normal * (2 * dot(norm_ray, normal));
  return CURR(compute_light)(
      vec3(1, 1, 1), //color
      vec3(1,1,1),
      {0, 1}, // material
      normal,
      norm_ray,
      ray_reflection,
      hit.point,
      false,
      distance_from_eye + hit.closest_point_distance_from_viewer_);
}


vec3 CURR(trace_all) (
    in vec3 norm_ray,
    in vec3 origin,
    float distance_from_eye) {
  assert(isfinite(distance_from_eye));
  vec3 pixel = vec3 (0.0, 0.0, 0.0);

  Hit hit = bbox_hit(norm_ray, origin);

  Hit light = light_hit(norm_ray, origin);
//  SineHit sine = sine_hit(norm_ray, origin);
//  if (sine.closest_point_distance_from_viewer_ < light.closest_point_distance_from_viewer_) {
//    if (sine.closest_point_distance_from_viewer_ <
//        hit.closest_point_distance_from_viewer_) {
//      return CURR(sine_trace)(sine, norm_ray, origin, distance_from_eye);
//    }
//  } else {
    if (light.closest_point_distance_from_viewer_ <
        hit.closest_point_distance_from_viewer_) {
      return light_trace(light, norm_ray, origin, distance_from_eye);
    }
//  }

  if (hit.id_ < 0) {
    return CURR(room_trace)(norm_ray, origin, distance_from_eye);
  }

  return CURR(ball_trace)(hit, norm_ray, origin, distance_from_eye);
}

vec3 CURR(trace) (
    in vec3 norm_ray,
    in vec3 origin,
    float distance_from_eye) {
  vec3 color = CURR(trace_all)(norm_ray, origin, distance_from_eye);
  return max(color, vec3(0));
}
