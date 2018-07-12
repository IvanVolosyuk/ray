#include <stdio.h>
#include <vector>
#include <thread>
#include <mutex>
#include <stdlib.h>
#include <condition_variable>

#include "ray.hpp"
#include "vector.hpp"

#include <SDL2/SDL.h>
#include <algorithm>
#include <random>
#include <functional>
#include <unistd.h>

using namespace std::placeholders;

//#define P(x) print(#x, x);
#define P(x) {}

using vec3 = Point;

struct Ball {
  Ball(const vec3 position, const vec3 color) : position_(position), color_(color) {}
  vec3 position_, color_;
};

struct Hit {
  Hit(int id, float c, float d) : id_(id), closest_point_distance_from_viewer_(c), distance_from_object_center2_(d) {}
  int id_;
  float closest_point_distance_from_viewer_;
  float distance_from_object_center2_;
};

float max_distance = 1000;
Hit no_hit = Hit(-1, max_distance, 0);

int window_width = 480;
int window_height = 270;
bool trace_values = false;

vec3 light_pos = vec3(-4.2, -3, 2);
float light_size = 0.9;
float light_power = 50.4;
vec3 light_color = vec3(light_power, light_power, light_power);
float light_size2 = light_size * light_size;
float light_inv_size = 1 / light_size;

float ball_size = 0.9;
float ball_size2 = ball_size * ball_size;
float ball_inv_size = 1 / ball_size;
std::vector<Ball> balls = {
  {vec3 (-1, -2, ball_size * 1.0f), vec3(1, 1, 1)},
  {vec3 (-2 * ball_size, 0, ball_size), vec3(0.01, 1.0, 0.01)},
  { vec3 (2 * ball_size, 0, ball_size), vec3(1.00, 0.00, 1.)},
};

vec3 floor_color = vec3 (0.14, 1.0, 0.14);
vec3 wall_color = vec3 (0.85, 0.8, 0.48);
vec3 ceiling_color = vec3 (0.98, 0.98, 0.98);
float defuse_attenuation = 0.4;

int max_depth = 2;

std::random_device rd;
std::mt19937 gen(rd());
float focused_distance = 1;
float lense_blur = 0.01;
std::normal_distribution<float> lense_gen{-lense_blur,lense_blur};
std::uniform_real_distribution<float> reflect_gen{0.f, 1.f};

std::normal_distribution<float> wall_gen{-0.8,0.8};
std::normal_distribution<float> light_gen{light_size, light_size};
std::normal_distribution<float> antialiasing{-0.5,0.5};

float room_size = 6;
float ceiling_z = room_size;
float wall_x0 = -room_size;
float wall_x1 = room_size;
float wall_y0 = -room_size;
float wall_y1 = room_size;

int max_internal_reflections = 30;

static unsigned long x=123456789, y=362436069, z=521288629;

unsigned long xorshf96(void) {          //period 2^96-1
  unsigned long t;
  x ^= x << 16;
  x ^= x >> 5;
  x ^= x << 1;

  t = x;
  x = y;
  y = z;
  z = t ^ x ^ y;

  return z;
}

vec3 distr() {
  return vec3(wall_gen(gen), wall_gen(gen), wall_gen(gen));
  long x = xorshf96();
  return vec3(
      (x & 0xFFFF) * (0.1/0x10000),
      ((x>>16) & 0xFFFF) * (0.1/0x10000),
      ((x>>24) & 0xFFFF) * (0.1/0x10000));
}

vec3 light_distr() {
  return vec3(light_gen(gen), light_gen(gen), light_gen(gen));
  long x = xorshf96();
  return vec3(
      (x & 0xFFFF) * (1./0x10000),
      ((x>>16) & 0xFFFF) * (1./0x10000),
      ((x>>24) & 0xFFFF) * (1./0x10000));
}

int numCPU = 0;

void print(const char* msg, const vec3 v) {
  if (trace_values)
    printf("%s: %f %f %f\n", msg, v.x, v.y, v.z);
}

void print(const char* msg, float v) {
  if (trace_values)
    printf("%s: %f\n", msg, v);
}

void print(const char*, const char* v) {
  if (trace_values)
    printf("%s\n", v);
}

void assert_norm(const vec3 v) {
  float s = v.size();
  assert(s > 0.99);
  assert(s < 1.01);
}

vec3 black = vec3(0, 0, 0);

vec3 trace(const vec3 norm_ray, const vec3 origin, int depth, float distance_from_eye);
vec3 compute_light(const vec3 color, const vec3 normal, const vec3 reflection,
    const vec3 point, bool rought_surface, int depth, float distance_from_eye);

// Glass refraction index
float glass_refraction_index = 1.492; //1.458;
float OBJECT_REFLECTIVITY = 0;

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
    cosX = sqrtf(1.0-sinT2);
  }
  float x = 1.0-cosX;
  float ret = r0+(1.0-r0)*x*x*x*x*x;

  // adjust reflect multiplier for object reflectivity
  ret = (OBJECT_REFLECTIVITY + (1.0-OBJECT_REFLECTIVITY) * ret);
  return ret;
}

Hit ball_hit(int id, const vec3 norm_ray, const vec3 origin) {
  P(norm_ray);
  vec3 ball_vector = balls[id].position_ - origin;

  float closest_point_distance_from_viewer = dot(norm_ray, ball_vector);
  if (closest_point_distance_from_viewer < 0) {
    return no_hit;
  }

  float ball_distance2 = ball_vector.size2();
  float distance_from_object_center2 = ball_distance2 -
    closest_point_distance_from_viewer * closest_point_distance_from_viewer;
  if (distance_from_object_center2 > ball_size2) {
    return no_hit;
  }
  return Hit(id, closest_point_distance_from_viewer, distance_from_object_center2);
}

vec3 light_trace(const Hit& p, vec3 norm_ray, vec3 origin, int depth, float distance_from_eye) {
  float distance_from_origin = p.closest_point_distance_from_viewer_ - sqrtf(ball_size2 - p.distance_from_object_center2_);
  vec3 intersection = origin + norm_ray * distance_from_origin;
  vec3 distance_from_light_vector = intersection - light_pos;

  vec3 normal = distance_from_light_vector * light_inv_size;
  float angle = -dot(norm_ray, normal);
  float total_distance = distance_from_eye + distance_from_origin;

  return light_color * (angle / (total_distance * total_distance));
}

class Hit light_hit(const vec3 norm_ray, const vec3 origin) {
  vec3 light_vector = light_pos - origin;
  float light_distance2 = light_vector.size2();

  float closest_point_distance_from_origin = dot(norm_ray, light_vector);
  if (closest_point_distance_from_origin < 0) {
    return no_hit;
  }

  P(closest_point_distance_from_origin);
  float distance_from_light_center2 = light_distance2 -
    closest_point_distance_from_origin * closest_point_distance_from_origin;
  if (distance_from_light_center2 > light_size2) {
    return no_hit;
  }
  return Hit(-2, closest_point_distance_from_origin, distance_from_light_center2);
}

vec3 trace_ball0_internal(const vec3 norm_ray, const vec3 origin, int depth, float distance_from_eye, int reflection) {
//  assert(distance_from_eye < 10000 && distance_from_eye >= 0);
  vec3 ball_vector = balls[0].position_ - origin;
  float closest_point_distance_from_viewer = dot(norm_ray, ball_vector);
  float ball_distance2 = ball_vector.size2();

  float distance_from_origin = 2 * closest_point_distance_from_viewer;
  vec3 intersection = origin + norm_ray * distance_from_origin;
  vec3 distance_from_ball_vector = intersection - balls[0].position_;
  vec3 normal = distance_from_ball_vector * ball_inv_size;

  if (FresnelReflectAmount(glass_refraction_index, 1, normal, norm_ray) > reflect_gen(gen)) {
    return vec3(0,0,0);
    vec3 ray_reflection = norm_ray + normal * (2 * dot(norm_ray, normal));
    if (reflection <= 0) return vec3();
    return trace_ball0_internal(ray_reflection, origin, depth, distance_from_eye + distance_from_origin, --reflection);
  } else {
    // refract
    float cosi = dot(normal, norm_ray);
    normal = -normal;
    float eta = glass_refraction_index;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    vec3 refracted_ray_norm = norm_ray * eta  + normal * (eta * cosi - sqrtf(k));
    return trace(refracted_ray_norm, intersection, depth, distance_from_eye + distance_from_origin);
  }
}

vec3 ball_trace(const Hit p, const vec3 norm_ray, const vec3 origin, int depth, float distance_from_eye) {
  float distance_from_origin = p.closest_point_distance_from_viewer_ - sqrtf(ball_size2 - p.distance_from_object_center2_);
  vec3 intersection = origin + norm_ray * distance_from_origin;

  P(intersection);
  vec3 distance_from_ball_vector = intersection - balls[p.id_].position_;

  vec3 normal = distance_from_ball_vector * ball_inv_size;

  auto make_reflection = [&]() {
    vec3 ray_reflection = norm_ray - normal * (2 * dot(norm_ray, normal));
    return compute_light(balls[p.id_].color_,
        normal,
        ray_reflection,
        intersection,
        false,
        depth,
        distance_from_eye + distance_from_origin);
  };

  auto make_refraction = [&]() {
    float cosi = -dot(normal, norm_ray);
    // FIXME: hack
    if (cosi < 0) return vec3();
    float eta = 1.f/glass_refraction_index;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    vec3 refracted_ray_norm = norm_ray * eta  + normal * (eta * cosi - sqrtf(k));
    return trace_ball0_internal(refracted_ray_norm, intersection, depth,
        distance_from_eye + distance_from_origin, max_internal_reflections);
  };

  if (p.id_ != 0) {
    return make_reflection();
  }

  float reflect_ammount = FresnelReflectAmount(1, glass_refraction_index, normal, norm_ray);
  if (reflect_ammount >= 1.f) {
    return make_reflection();
  }

  if (depth == max_depth) {
    // Trace both if first ray
    return make_reflection() * reflect_ammount + make_refraction() * (1 - reflect_ammount);
  }
  if (reflect_ammount > reflect_gen(gen)) {
    return make_reflection();
  } else {
    return make_refraction();
  }
}

struct RoomHit {
  float min_dist;
  vec3 normal;
  vec3 reflection;
  vec3 color;
};

RoomHit room_hit(const vec3 norm_ray, const vec3 point) {
//  vec3 room_a(wall_x0, wall_y0, 0);
//  vec3 room_b(wall_x1, wall_y1, ceiling_z);
//  vec3 tMin = (room_a - point) / norm_ray;
//  vec3 tMax = (room_b - point) / norm_ray;
//  vec3 t1 = min(dist_a, dist_b);
//  vec3 t2 = max(dist_a, dist_b);
//  float tNear = max(max(t1.x, t1.y), t1.z);
//  float tFar = min(min(t2.x, t2.y), t2.z);

  float dist_x, dist_y, dist_z;
  if (norm_ray.z >= 0) {
    // tracng ceiling
    dist_z = (ceiling_z-point.z) / norm_ray.z;
  } else {
    // tracing floor
    dist_z = (/*0*/-point.z) / norm_ray.z;
  }
  if (norm_ray.x >= 0) {
    // tracing ceiling
    dist_x = (wall_x1-point.x) / norm_ray.x;
  } else {
    // tracing floor
    dist_x = (wall_x0-point.x) / norm_ray.x;
  }

  if (norm_ray.y >= 0) {
    // tracing ceiling
    dist_y = (wall_y1-point.y) / norm_ray.y;
  } else {
    // tracing floor
    dist_y = (wall_y0-point.y) / norm_ray.y;
  }
  vec3 normal;
  vec3 reflection;
  float min_dist;
  vec3 color;
  if (dist_y < dist_z) {
    color = wall_color;
    if (dist_x < dist_y) {
      min_dist = dist_x;
      reflection = vec3(-norm_ray.x, norm_ray.y, norm_ray.z);
      normal = vec3(std::copysign(1, reflection.x), 0, 0);
    } else {
      min_dist = dist_y;
      reflection = vec3(norm_ray.x, -norm_ray.y, norm_ray.z);
      normal = vec3(0, std::copysign(1, reflection.y), 0);
    }
  } else {
    if (dist_x < dist_z) {
      min_dist = dist_x;
      reflection = vec3(-norm_ray.x, norm_ray.y, norm_ray.z);
      normal = vec3(std::copysign(1, reflection.x), 0, 0);
      color = wall_color;
    } else {
      min_dist = dist_z;
      reflection = vec3(norm_ray.x, norm_ray.y, -norm_ray.z);
      normal = vec3(0, 0, std::copysign(1, reflection.z));
      color = std::signbit(reflection.z) ? ceiling_color : floor_color;
    }
  }
  return {min_dist, normal, reflection, color};
}

vec3 trace_room(const vec3 norm_ray, const vec3 point, int depth, float distance_from_eye) {
  RoomHit p = room_hit(norm_ray, point);
  vec3 ray = norm_ray * p.min_dist;
  vec3 intersection = point + ray;
  // tiles
  vec3 color = p.color;
  if (intersection.z < 0.01) {
    color = ((int)(intersection.x + 10) % 2 == (int)(intersection.y + 10) % 2) ? vec3(0.1, 0.1, 0.1) : vec3(1,1,1);
  }
  return compute_light(color, p.normal, p.reflection, intersection, true, depth, distance_from_eye + p.min_dist);
}

vec3 trace(const vec3 norm_ray, const vec3 origin, int depth, float distance_from_eye) {
  Hit hit = no_hit;
  for (int i = 0; i < balls.size(); i++) {
    Hit other_hit = ball_hit(i, norm_ray, origin);
    if (other_hit.closest_point_distance_from_viewer_ < hit.closest_point_distance_from_viewer_) {
      hit = other_hit;
    }
  }
  Hit light = light_hit(norm_ray, origin);
  if (light.closest_point_distance_from_viewer_ < hit.closest_point_distance_from_viewer_) {
    return light_trace(light, norm_ray, origin, depth, distance_from_eye);
  }
  if (hit.id_ >= 0) {
    return ball_trace(hit, norm_ray, origin, depth, distance_from_eye);
  }
  return trace_room(norm_ray, origin, depth, distance_from_eye);
}

float distance(const vec3 norm_ray, const vec3 origin) {
  Hit hit = no_hit;
  for (int i = 0; i < balls.size(); i++) {
    Hit another_hit = ball_hit(i, norm_ray, origin);
    if (another_hit.closest_point_distance_from_viewer_
           < hit.closest_point_distance_from_viewer_) {
      hit = another_hit;
    }
  }
  Hit light = light_hit(norm_ray, origin);
  if (light.closest_point_distance_from_viewer_ <
              hit.closest_point_distance_from_viewer_) {
    return light.closest_point_distance_from_viewer_
      - sqrtf(light_size2 - light.distance_from_object_center2_);
  }

  if (hit.id_ >= 0) {
    float distance_from_origin = hit.closest_point_distance_from_viewer_
      - sqrtf(ball_size2 - hit.distance_from_object_center2_);
    return distance_from_origin;
  }

  RoomHit rt = room_hit(norm_ray, origin);
  P(rt.min_dist);
  return rt.min_dist;
}

vec3 compute_light(
    const vec3 color,
    const vec3 normal,
    const vec3 reflection_in,
    const vec3 point,
    bool rought_surface,
    int depth,
    float distance_from_eye) {
  vec3 total_color = black;
  if (depth > 0) {
    vec3 reflection = reflection_in;
    if (rought_surface) {
      reflection = (reflection + distr()).normalize();
    }
    vec3 second_ray = trace(reflection, point, depth - 1, distance_from_eye);
    total_color = color * second_ray * defuse_attenuation;
  }
  if (!rought_surface) {
    return total_color;
  }

  vec3 light_rnd_pos = light_pos + light_distr();
  vec3 light_from_point = light_rnd_pos - point;
  float angle_x_distance = dot(normal, light_from_point);
  if (angle_x_distance < 0) {
    return total_color;
  }
  float light_distance2 = light_from_point.size2();
  float light_distance_inv = 1/sqrtf(light_distance2);
  float light_distance = 1/light_distance_inv;
  vec3 light_from_point_norm = light_from_point * light_distance_inv;

  for (int i = 0; i < balls.size(); i++) {
    Hit hit = ball_hit(i, light_from_point_norm, point);
    if (hit.closest_point_distance_from_viewer_ < light_distance) {
      // Obstracted
      return total_color;
    }
  }

  float angle = angle_x_distance * light_distance_inv;
  float total_distance = light_distance + distance_from_eye;
  vec3 defuse_color = (color * light_color) * (angle / (total_distance * total_distance) * defuse_attenuation);
  total_color += defuse_color;
  return total_color;
}


vec3 sight = vec3(0., 1, -0.1).normalize();
vec3 sight_x, sight_y;

// Sort objects by distance / obstraction possibility
Uint8 colorToInt(float c) {
  if (c > 1) return 255;
  return sqrtf(c) * 255;
}
vec3 saturateColor(vec3 c) {
  float m = std::max(c.x, c.y);
  if (m < 1) {
    return c;
  }
  float total = c.x + c.y + c.z;
  if (total > 3) {
    return vec3(1,1,1);
  }
  float scale = (3 - total) / (3 * m - total);
  float grey = 1 - scale * m;
  return vec3(grey + scale * c.x,
                grey + scale * c.y,
                grey + scale * c.z);
}


void check_saturation() {
  auto s = saturateColor(vec3(2, 0.1, 0));
  P(s);
  assert(s.sum() == vec3(1, 0.6, 0.5).sum());
}

vec3 viewer = vec3(0, -5.5, 1.5);

std::vector<std::thread> threads;
std::mutex m;
std::condition_variable cv;
int frame = 0;
int base_frame = 0;
bool die = false;
int num_running = 0;
BasePoint<double>* fppixels;
Uint8* pixels;

void set_focus_distance(float x, float y) {
  vec3 yoffset = sight_y * (float)(window_height / 2);
  vec3 xoffset = sight_x * (float)(window_width / 2);

  vec3 ray = sight - yoffset - xoffset + sight_y * y + sight_x * x;
  P(ray);
  vec3 norm_ray = ray.normalize();
  P(norm_ray);
  focused_distance = distance(norm_ray, viewer);
}

void drawThread(int id) {
  vec3 yoffset = sight_y * (float)(window_height / 2);
  vec3 xoffset = sight_x * (float)(window_width / 2);
  Uint8* my_pixels = pixels;
  BasePoint<double>* my_fppixels = fppixels;
  int num_frames = frame - base_frame;
  if ((num_frames & (num_frames - 1)) == 0 && id == 0 && num_frames > 16) {
    printf("Num frames: %d\n", num_frames);
  }
  double one_mul = 1. / num_frames;

  vec3 yray = sight - yoffset - xoffset;
  for (int y = 0; y < window_height; y++) {
    if (y % numCPU == id) {
      vec3 ray = yray;
      for (int x = 0; x < window_width; x++) {
        vec3 focused_ray = (ray + sight_x * antialiasing(gen) + sight_y * antialiasing(gen)).normalize();
        vec3 focused_point = viewer + focused_ray * focused_distance;
        vec3 me = viewer + sight_x.normalize() * (float)lense_gen(gen) + sight_y.normalize() * (float)lense_gen(gen);
        vec3 new_ray = (focused_point - me).normalize();

        trace_values = x == 500 && y == 500;
        auto res = trace(new_ray, me, max_depth, 0);
        // accumulate
        *my_fppixels += BasePoint<double>::convert(res);
        res = BasePoint<float>::convert(*my_fppixels++ * one_mul);

        vec3 saturated = saturateColor(res);
        *my_pixels++ = colorToInt(saturated.x);
        *my_pixels++ = colorToInt(saturated.y);
        *my_pixels++ = colorToInt(saturated.z);
        *my_pixels++ = 255;
        ray += sight_x;
      }
    } else {
      my_pixels += window_width * 4;
      my_fppixels += window_width;
    }
    yray += sight_y;
  }
}

void worker(int id) {
  int my_frame = 0;
  bool my_die;
  while (true) {
    my_frame++;
    {
      std::unique_lock<std::mutex> lk(m);
      cv.wait(lk, [&my_frame]{return my_frame == frame;});
      my_die = die;
    }
    if (!my_die) {
      drawThread(id);
    }
    {
      std::unique_lock<std::mutex> lk(m);
      num_running--;
      cv.notify_all();
      if (my_die) return;
    }
  }
}

void draw() {
  if (threads.size() == 0) {
    for (int i = 0; i < numCPU; i++) {
      threads.push_back(std::thread(worker, i));
    }
  }

  {
    std::unique_lock<std::mutex> lk(m);
    num_running = threads.size();
    // Clear accumulated image
    if (frame == base_frame) {
      memset(fppixels, 0, window_width * window_height * sizeof(BasePoint<double>));
    }
    frame++;
    cv.notify_all();
  }
  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, []{return num_running == 0;});
  }
}

void reset_accumulate() {
  base_frame = frame;
}

void update_viewpoint() {
  float dx = 1.9 / window_width;
  sight_x = cross(sight, vec3(0,0,1)).normalize();
  sight_y = cross(sight, sight_x);
  sight_x *= dx;
  sight_y *= dx;
}

int main(void) {
    const char* cpus = getenv("NUM_CPUS");
    if (cpus != nullptr) {
      numCPU = atoi(cpus);
    }
    if (numCPU == 0) {
      numCPU = sysconf(_SC_NPROCESSORS_ONLN);
    }
    printf("Num CPUs: %d\n", numCPU);
    check_saturation();
    SDL_Event event;
    SDL_Renderer *renderer;
    SDL_Window *window;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(window_width, window_height, 0, &window, &renderer);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
    SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
    SDL_Texture * texture = SDL_CreateTexture(renderer,
                SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC, window_width, window_height);
    fppixels = new BasePoint<double>[window_width * window_height];
    pixels = new Uint8[window_width * window_height * 4];

    SDL_RenderPresent(renderer);
    Uint32 ts_move_forward;
    Uint32 ts_move_backward;
    Uint32 ts_strafe_left;
    Uint32 ts_strafe_right;

    bool relative_motion = false;
    update_viewpoint();
    int mouse_x_before_rel = 0;
    int mouse_y_before_rel = 0;

    auto apply_motion = [](vec3 dir, Uint32* prev_ts, Uint32 ts) {
      if (*prev_ts == 0) {
        return;
      }
      if (ts < *prev_ts) {
        return;
      }
      dir.z = 0;
      int dt = ts - *prev_ts;
      viewer += dir * (0.001f * dt);
      *prev_ts = ts;
      reset_accumulate();
    };

    while (1) {
      auto move_forward = std::bind(apply_motion, sight, &ts_move_forward, _1);
      auto move_backward = std::bind(apply_motion, -sight, &ts_move_backward, _1);
      auto move_left = std::bind(apply_motion, -sight_x.normalize(), &ts_strafe_left, _1);
      auto move_right = std::bind(apply_motion, sight_x.normalize(), &ts_strafe_right, _1);

      while(SDL_PollEvent(&event)) {
        switch (event.type) {
          case SDL_QUIT:
            goto exit;
            break;
          case SDL_MOUSEBUTTONUP:
                         if (event.button.button == SDL_BUTTON_RIGHT) {
                           SDL_SetRelativeMouseMode(SDL_FALSE);
//                           SDL_WarpMouseInWindow(window,
//                               mouse_x_before_rel,
//                               mouse_y_before_rel);
                           relative_motion = false;
                         }
                         break;
          case SDL_MOUSEBUTTONDOWN:
                         if (event.button.button == SDL_BUTTON_RIGHT) {
                           SDL_SetRelativeMouseMode(SDL_TRUE);
                           SDL_SetWindowGrab(window, SDL_TRUE);
                           relative_motion = true;
                           mouse_x_before_rel = event.button.x;
                           mouse_y_before_rel = event.button.y;
                         }
                         if (event.button.button == SDL_BUTTON_LEFT) {
                           if (relative_motion) {
                             printf("Focus center\n");
                             set_focus_distance(window_width / 2, window_height / 2);
                           } else {
                             set_focus_distance(event.button.x, event.button.y);
                           }
                           printf("Focused distance: %f\n", focused_distance);
                           reset_accumulate();
                         }
                         break;
          case SDL_MOUSEMOTION:
                         if (event.motion.state == SDL_BUTTON_RMASK) {
                           vec3 x = cross(sight, vec3(0.f,0.f,1.f)).normalize();
                           vec3 y = cross(sight, x).normalize();
                           sight = (cross(sight, y) * (-0.001f * event.motion.xrel) + sight).normalize();
                           sight = (cross(sight, x) * (0.001f * event.motion.yrel) + sight).normalize();
                           update_viewpoint();
                           reset_accumulate();
                         }
                         break;
          case SDL_KEYUP:
          case SDL_KEYDOWN:
                         auto update_key_ts = [&](Uint32* state) {
//                           printf("State %d ts = %d\n", event.key.state, event.key.timestamp);
                           *state = (event.key.state == SDL_PRESSED) ? event.key.timestamp : 0;
                         };

                         switch (event.key.keysym.scancode) {
                           case SDL_SCANCODE_ESCAPE:
                             die = true;
                             draw();
                             for (auto& t : threads) t.join();
                             goto exit;
                           case SDL_SCANCODE_W:
                             move_forward(event.key.timestamp);
                             update_key_ts(&ts_move_forward);
                             break;
                           case SDL_SCANCODE_S:
                             move_backward(event.key.timestamp);
                             update_key_ts(&ts_move_backward);
                             break;
                           case SDL_SCANCODE_A:
                             move_left(event.key.timestamp);
                             update_key_ts(&ts_strafe_left);
                             break;
                           case SDL_SCANCODE_E:
                           case SDL_SCANCODE_D:
                             move_right(event.key.timestamp);
                             update_key_ts(&ts_strafe_right);
                             break;
                           case SDL_SCANCODE_1:
                             max_depth = 0;
                             if (event.key.state != SDL_PRESSED) reset_accumulate();
                             break;
                           case SDL_SCANCODE_2:
                             max_depth = 1;
                             if (event.key.state != SDL_PRESSED) reset_accumulate();
                             break;
                           case SDL_SCANCODE_3:
                             max_depth = 2;
                             if (event.key.state != SDL_PRESSED) reset_accumulate();
                             break;
                           case SDL_SCANCODE_4:
                             max_depth = 3;
                             if (event.key.state != SDL_PRESSED) reset_accumulate();
                             break;
                           case SDL_SCANCODE_7:
                             wall_gen = std::normal_distribution<float>{-0.00,0.00};
                             if (event.key.state != SDL_PRESSED) reset_accumulate();
                             break;
                           case SDL_SCANCODE_8:
                             wall_gen = std::normal_distribution<float>{-0.03,0.03};
                             if (event.key.state != SDL_PRESSED) reset_accumulate();
                             break;
                           case SDL_SCANCODE_9:
                             wall_gen = std::normal_distribution<float>{-0.8,0.8};
                             if (event.key.state != SDL_PRESSED) reset_accumulate();
                             break;
                           case SDL_SCANCODE_MINUS:
                             printf("Light size 0.1\n");
                             light_size = 0.1;
                             light_size2 = light_size * light_size;
                             light_inv_size = 1 / light_size;
                             light_gen = std::normal_distribution<float>{light_size, light_size};
                             if (event.key.state != SDL_PRESSED) reset_accumulate();
                             break;
                           case SDL_SCANCODE_EQUALS:
                             printf("Light size 2\n");
                             light_size = 0.9;
                             light_size2 = light_size * light_size;
                             light_inv_size = 1 / light_size;
                             light_gen = std::normal_distribution<float>{light_size, light_size};
                             if (event.key.state != SDL_PRESSED) reset_accumulate();
                             break;
                           case SDL_SCANCODE_LEFTBRACKET:
                             lense_blur = std::max(0.f, lense_blur * 0.8f - .0001f);
                             if (lense_blur == 0) {
                               printf("No blur\n");
                             }
                             lense_gen = std::normal_distribution<float>{-lense_blur,lense_blur};
                             if (event.key.state != SDL_PRESSED) reset_accumulate();
                             break;

                           case SDL_SCANCODE_RIGHTBRACKET:
                             lense_blur = lense_blur * 1.2f + .0001f;
                             lense_gen = std::normal_distribution<float>{-lense_blur,lense_blur};
                             if (event.key.state != SDL_PRESSED) reset_accumulate();
                             break;
                           case SDL_SCANCODE_F:
                             if (event.key.state != SDL_PRESSED) {
                               SDL_DestroyRenderer(renderer);
                               SDL_DestroyWindow(window);
                               SDL_DestroyTexture(texture);
                               delete fppixels;
                               delete pixels;

                               if (window_width == 480) {
                                 window_width = 1920;
                                 window_height = 1080;
                               } else {
                                 window_width = 480;
                                 window_height = 270;
                               }

                               SDL_CreateWindowAndRenderer(window_width, window_height, 0, &window, &renderer);
                               texture = SDL_CreateTexture(renderer,
                                   SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC, window_width, window_height);
                               fppixels = new BasePoint<double>[window_width * window_height];
                               pixels = new Uint8[window_width * window_height * 4];
                               update_viewpoint();
                               reset_accumulate();
                             }
                             break;
                           case SDL_SCANCODE_Z:
                             if (event.key.state != SDL_PRESSED)
                               SDL_SetWindowGrab(window, SDL_FALSE);
                         }
        }
      }

      Uint32 newTime = SDL_GetTicks();
      move_forward(newTime);
      move_backward(newTime);
      move_left(newTime);
      move_right(newTime);

      draw();
      SDL_UpdateTexture(texture, NULL, (void*)pixels, window_width * sizeof(Uint32));
      SDL_RenderCopy(renderer, texture, NULL, NULL);
      SDL_RenderPresent(renderer);
      event.type = -1;
    }
exit:
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return EXIT_SUCCESS;
}
