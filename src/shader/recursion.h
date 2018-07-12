#define RECURSIVE(name, stage) name ## _ ## stage

vec3 CURR(compute_light) (
    in vec3 color,
    in vec3 normal,
    in vec3 reflection_in,
    in vec3 point,
    bool rought_surface,
    in float distance_from_eye) {
  vec3 total_color = black;

#ifdef NEXT
  vec3 reflection = reflection_in;
  if (rought_surface) {
    reflection = normalize(reflection + wall_distr(point));
  }
  vec3 second_ray = NEXT(trace)(reflection, point, distance_from_eye);
  total_color = (color * second_ray) * defuse_attenuation;
#endif

  if (!rought_surface) {
    return total_color;
  }

  vec3 light_rnd_pos = light_pos + light_distr(point);
  vec3 light_from_point = light_rnd_pos - point;
  float angle_x_distance = dot(normal, light_from_point);
  if (angle_x_distance < 0) {
    return total_color;
  }

  float light_distance2 = dot(light_from_point, light_from_point);
  float light_distance_inv = inversesqrt(light_distance2);
  float light_distance = 1.f/light_distance_inv;
  vec3 light_from_point_norm = light_from_point * light_distance_inv;

  for (int i = 0; i < balls.length(); i++) {
    Hit res = ball_hit(i, light_from_point_norm, point);
    if (res.closest_point_distance_from_viewer_ < light_distance) {
      // Obstracted
      return total_color;
    }
  }

  float angle = angle_x_distance * light_distance_inv;
  float total_distance = light_distance + distance_from_eye;
  vec3 defuse_color = (color * light_color) *
    (angle / (total_distance * total_distance) * defuse_attenuation);
  total_color += defuse_color;
  return total_color;
}

vec3 CURR(trace_ball0_internal)(
    in vec3 norm_ray,
    in vec3 origin,
    in float distance_from_eye) {
#ifdef NEXT
  for (int i = 0; i < max_internal_reflections; i++) {
    vec3 ball_vector = balls[0].position_ - origin;
    float closest_point_distance_from_viewer = dot(norm_ray, ball_vector);
    float ball_distance2 = dot(ball_vector, ball_vector);

    float distance_from_origin = 2 * closest_point_distance_from_viewer;
    vec3 intersection = origin + norm_ray * distance_from_origin;
    vec3 distance_from_ball_vector = intersection - balls[0].position_;
    vec3 normal = distance_from_ball_vector * ball_inv_size;

    if (FresnelReflectAmount(glass_refraction_index, 1, normal, norm_ray) > reflect_gen(origin)) {
      vec3 ray_reflection = norm_ray - normal * (2 * dot(norm_ray, normal));
      // Restart from new point
      norm_ray = ray_reflection;
      origin = intersection;
      distance_from_eye += distance_from_origin;
      continue;
    } else {
      // refract
      float cosi = dot(normal, norm_ray);
      normal = -normal;
      float eta = glass_refraction_index;
      float k = 1 - eta * eta * (1 - cosi * cosi);
      vec3 refracted_ray_norm = norm_ray * eta  + normal * (eta * cosi - sqrt(k));
      return NEXT(trace)(refracted_ray_norm, intersection, distance_from_eye + distance_from_origin);
    }
  }
#endif
  return vec3(0);
}

vec3 CURR(make_reflection)(
    vec3 color,
    vec3 norm_ray,
    vec3 normal,
    vec3 intersection,
    float distance_from_eye) {
  // FIXME: use reflect
  vec3 ray_reflection = norm_ray - normal * (2 * dot(norm_ray, normal));
  return CURR(compute_light)(
      color,
      normal,
      ray_reflection,
      intersection,
      false,
      distance_from_eye);
};

vec3 CURR(make_refraction)(
    vec3 norm_ray,
    vec3 normal,
    vec3 intersection,
    float distance_from_eye) {
  // FIXME: use refract
  float cosi = -dot(normal, norm_ray);
  // FIXME: hack
  if (cosi < 0) return vec3(0);
  float eta = 1.f/glass_refraction_index;
  float k = 1 - eta * eta * (1 - cosi * cosi);
  vec3 refracted_ray_norm = norm_ray * eta  + normal * (eta * cosi - sqrt(k));
  return CURR(trace_ball0_internal)(
      refracted_ray_norm,
      intersection,
      distance_from_eye);
};

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
    return CURR(make_reflection)(ball.color_, norm_ray, normal, intersection, total_distance);
  }

  float reflect_ammount = FresnelReflectAmount(1, glass_refraction_index, normal, norm_ray);

  if (reflect_ammount >= 1.f) {
    return CURR(make_reflection)(ball.color_, norm_ray, normal, intersection, total_distance);
  }


#ifdef MAX_STAGE
    // Trace both if first ray
    return CURR(make_reflection)(
        ball.color_,
        norm_ray, normal,
        intersection,
        total_distance) * reflect_ammount +
      CURR(make_refraction)(
          norm_ray,
          normal,
          intersection,
          total_distance) * (1 - reflect_ammount);
#else
  if (reflect_ammount > reflect_gen(origin)) {
    return CURR(make_reflection)(ball.color_, norm_ray, normal, intersection, total_distance);
  } else {
    return CURR(make_refraction)(norm_ray, normal, intersection, total_distance);
  }
#endif
}

vec3 CURR(room_trace) (
    in vec3 norm_ray,
    in vec3 origin,
    float distance_from_eye) {
  RoomHit p = room_hit(norm_ray, origin);
  vec3 ray = norm_ray * p.min_dist;
  vec3 intersection = origin + ray;
  // tiles
  vec3 color = p.color;
  if (intersection.z < 0.01) {
    color = fract(
        (floor(intersection.x + 10) +
         floor(intersection.y + 10)) * 0.5) == 0 ? vec3(0.1, 0.1, 0.1) : vec3(1,1,1);
  }

  return CURR(compute_light)(
      color, p.normal,
      p.reflection,
      intersection,
      true,
      distance_from_eye + p.min_dist);
}

vec3 CURR(trace) (
    in vec3 norm_ray,
    in vec3 origin,
    float distance_from_eye) {
  vec3 pixel = vec3 (0.0, 0.0, 0.0);
  Hit hit = no_hit;

  for (int i = 0; i < balls.length(); i++) {
    Hit other_hit = ball_hit(i, norm_ray, origin);
    if (other_hit.closest_point_distance_from_viewer_ <
        hit.closest_point_distance_from_viewer_) {
      hit = other_hit;
    }
  }

  Hit light = light_hit(norm_ray, origin);
  if (light.closest_point_distance_from_viewer_ <
      hit.closest_point_distance_from_viewer_) {
    return light_trace(light, norm_ray, origin, distance_from_eye);
  }

  if (hit.id_ < 0) {
    return CURR(room_trace)(norm_ray, origin, distance_from_eye);
  }

  return CURR(ball_trace)(hit, norm_ray, origin, distance_from_eye);
}