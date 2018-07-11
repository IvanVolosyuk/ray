#include <stdio.h>
#include <vector>
#include <thread>
#include <mutex>
#include <stdlib.h>
#include <condition_variable>

#include "ray.hpp"
#include "vector.hpp"

#include <SDL2/SDL.h>
#include <boost/optional.hpp>
#include <algorithm>
#include <random>

//#define P(x) print(#x, x);
#define P(x) {}


using Vector = Point;
#define optional boost::optional
#define nullopt boost::none

int window_width = 480;
int window_height = 270;

Point light_pos = Point(-4.2, -3, 2);
float light_size = 0.9;
float light_power = 50.4;
Point light_color = Point(light_power, light_power, light_power);
float light_size2 = light_size * light_size;
float light_inv_size = 1 / light_size;

float ball_size = 0.9;
float ball_size2 = ball_size * ball_size;
float ball_inv_size = 1 / ball_size;
bool trace_values = false;

Vector floor_color(0.14, 1.0, 0.14);
Vector wall_color(0.85, 0.8, 0.48);
Vector ceiling_color(0.98, 0.98, 0.98);
Vector floor_normal(0, 0, 1);
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

float room_size = 6;
float ceiling_z = room_size;
float wall_x0 = -room_size;
float wall_x1 = room_size;
float wall_y0 = -room_size;
float wall_y1 = room_size;

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

Vector distr() {
  return Vector(wall_gen(gen), wall_gen(gen), wall_gen(gen));
  long x = xorshf96();
  return Vector(
      (x & 0xFFFF) * (0.1/0x10000),
      ((x>>16) & 0xFFFF) * (0.1/0x10000),
      ((x>>24) & 0xFFFF) * (0.1/0x10000));
}

Vector light_distr() {
  return Vector(light_gen(gen), light_gen(gen), light_gen(gen));
  long x = xorshf96();
  return Vector(
      (x & 0xFFFF) * (1./0x10000),
      ((x>>16) & 0xFFFF) * (1./0x10000),
      ((x>>24) & 0xFFFF) * (1./0x10000));
}

int numCPU = 0;

void print(const char* msg, const Vector& v) {
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

void assert_norm(const Vector& v) {
  float s = v.size();
  assert(s > 0.99);
  assert(s < 1.01);
}

Point black = Point(0, 0, 0);

optional<Point> trace(const Vector& norm_ray, const Vector& origin, int depth, float distance_from_eye);
Point compute_light(const Point& color, const Vector& normal, const Vector& reflection,
    const Point& point, bool rought_surface, int depth, float distance_from_eye);

class Ball;

class Hit {
  public:
    const Ball *ball_;
    float closest_point_distance_from_viewer_;
    float distance_from_object_center2_;
};

// Glass refraction index
float glass_refraction_index = 1.492; //1.458;
float OBJECT_REFLECTIVITY = 0;

float FresnelReflectAmount (float n1, float n2, Vector normal, Vector incident) {
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

class Ball {
  public:
    Ball(const Vector& position, const Vector& color) : position_(position), color_(color) {}

    optional<Hit> pretrace(const Vector& norm_ray, const Vector& origin) const {
      P(norm_ray);
      Vector ball_vector = position_ - origin;

      float closest_point_distance_from_viewer = dot(norm_ray, ball_vector);
      if (closest_point_distance_from_viewer < 0) {
        return nullopt;
      }

      float ball_distance2 = ball_vector.size2();
      float distance_from_object_center2 = ball_distance2 -
        closest_point_distance_from_viewer * closest_point_distance_from_viewer;
      if (distance_from_object_center2 > ball_size2) {
        return nullopt;
      }
      return Hit{this, closest_point_distance_from_viewer, distance_from_object_center2};
    }

    Vector position_, color_;
};

Point light_trace(const Hit& p, Vector norm_ray, Point origin, int depth, float distance_from_eye) {
  float distance_from_origin = p.closest_point_distance_from_viewer_ - sqrtf(ball_size2 - p.distance_from_object_center2_);
  Point intersection = origin + norm_ray * distance_from_origin;
  Vector distance_from_light_vector = intersection - light_pos;

  Vector normal = distance_from_light_vector * light_inv_size;
  float angle = -dot(norm_ray, normal);

  return light_color * (angle / (distance_from_eye * distance_from_eye));
}

class optional<Hit> pretrace_light(const Vector& norm_ray, const Vector& origin) {
  Vector light_vector = light_pos - origin;
  float light_distance2 = light_vector.size2();

  float closest_point_distance_from_origin = dot(norm_ray, light_vector);
  if (closest_point_distance_from_origin < 0) {
    return nullopt;
  }

  P(closest_point_distance_from_origin);
  float distance_from_light_center2 = light_distance2 -
    closest_point_distance_from_origin * closest_point_distance_from_origin;
  if (distance_from_light_center2 > light_size2) {
    return nullopt;
  }
  return Hit{nullptr, closest_point_distance_from_origin, distance_from_light_center2};
}

std::vector<Ball> balls = {
  {{-1, -2, ball_size * 1.0f}, {1, 1, 1}},
  {{-2 * ball_size, 0, ball_size}, {0.01, 1.0, 0.01}},
  {{2 * ball_size, 0, ball_size}, {1.00, 0.00, 1.}},
};

int max_internal_reflections = 30;

Point trace_ball0_internal(const Vector& norm_ray, const Vector& origin, int depth, float distance_from_eye, int reflection) {
//  assert(distance_from_eye < 10000 && distance_from_eye >= 0);
  Vector ball_vector = balls[0].position_ - origin;
  float closest_point_distance_from_viewer = dot(norm_ray, ball_vector);
  float ball_distance2 = ball_vector.size2();

  float distance_from_origin = 2 * closest_point_distance_from_viewer;
  Point intersection = origin + norm_ray * distance_from_origin;
  Vector distance_from_ball_vector = intersection - balls[0].position_;
  Vector normal = distance_from_ball_vector * ball_inv_size;

  if (FresnelReflectAmount(glass_refraction_index, 1, normal, norm_ray) > reflect_gen(gen)) {
    return Point(0,0,0);
    Vector ray_reflection = norm_ray + normal * (2 * dot(norm_ray, normal));
    if (reflection <= 0) return Point();
    return trace_ball0_internal(ray_reflection, origin, depth, distance_from_eye + distance_from_origin, --reflection);
  } else {
    // refract
    float cosi = dot(normal, norm_ray);
    normal = -normal;
    float eta = glass_refraction_index;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    Vector refracted_ray_norm = norm_ray * eta  + normal * (eta * cosi - sqrtf(k));
    auto res = trace(refracted_ray_norm, intersection, depth, distance_from_eye + distance_from_origin);
    if (!res) return Vector(0,0,0);
    return *res;
  }
}

Point ball_trace(const Hit& p, const Vector& norm_ray, const Vector& origin, int depth, float distance_from_eye) {
  float distance_from_origin = p.closest_point_distance_from_viewer_ - sqrtf(ball_size2 - p.distance_from_object_center2_);
  Point intersection = origin + norm_ray * distance_from_origin;

  P(intersection);
  Vector distance_from_ball_vector = intersection - p.ball_->position_;

  Vector normal = distance_from_ball_vector * ball_inv_size;
  if (p.ball_->position_ != balls[0].position_
      || FresnelReflectAmount(1, glass_refraction_index, normal, norm_ray) > reflect_gen(gen)) {
    Vector ray_reflection = norm_ray - normal * (2 * dot(norm_ray, normal));

    P(ray_reflection);
    return compute_light(p.ball_->color_,
        normal,
        ray_reflection,
        intersection,
        false,
        depth,
        distance_from_eye + distance_from_origin);
  } else {
    // refract
    float cosi = -dot(normal, norm_ray);
    // FIXME: hack
    if (cosi < 0) return Point(0,0,0);
    float eta = 1.f/glass_refraction_index;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    Vector refracted_ray_norm = norm_ray * eta  + normal * (eta * cosi - sqrtf(k));
    return trace_ball0_internal(refracted_ray_norm, intersection, depth,
        distance_from_eye + distance_from_origin, max_internal_reflections);
  }
}

struct RoomHit {
  float min_dist;
  Vector normal;
  Vector reflection;
  Vector color;
};

RoomHit pretrace_room(const Vector& norm_ray, const Vector& point) {
//  Point room_a(wall_x0, wall_y0, 0);
//  Point room_b(wall_x1, wall_y1, ceiling_z);
//  Point tMin = (room_a - point) / norm_ray;
//  Point tMax = (room_b - point) / norm_ray;
//  Point t1 = min(dist_a, dist_b);
//  Point t2 = max(dist_a, dist_b);
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
  Vector normal;
  Vector reflection;
  float min_dist;
  Vector color;
  if (dist_y < dist_z) {
    color = wall_color;
    if (dist_x < dist_y) {
      min_dist = dist_x;
      reflection = Vector(-norm_ray.x, norm_ray.y, norm_ray.z);
      normal = Vector(std::copysign(1, reflection.x), 0, 0);
    } else {
      min_dist = dist_y;
      reflection = Vector(norm_ray.x, -norm_ray.y, norm_ray.z);
      normal = Vector(0, std::copysign(1, reflection.y), 0);
    }
  } else {
    if (dist_x < dist_z) {
      min_dist = dist_x;
      reflection = Vector(-norm_ray.x, norm_ray.y, norm_ray.z);
      normal = Vector(std::copysign(1, reflection.x), 0, 0);
      color = wall_color;
    } else {
      min_dist = dist_z;
      reflection = Vector(norm_ray.x, norm_ray.y, -norm_ray.z);
      normal = Vector(0, 0, std::copysign(1, reflection.z));
      color = std::signbit(reflection.z) ? ceiling_color : floor_color;
    }
  }
  return {min_dist, normal, reflection, color};
}

optional<Point> trace_room(const Vector& norm_ray, const Vector& point, int depth, float distance_from_eye) {
  RoomHit p = pretrace_room(norm_ray, point);
  Point ray = norm_ray * p.min_dist;
  Point intersection = point + ray;
  // tiles
  Vector color = p.color;
  if (intersection.z < 0.01) {
    color = ((int)(intersection.x + 10) % 2 == (int)(intersection.y + 10) % 2) ? Vector(0.1, 0.1, 0.1) : Vector(1,1,1);
  }
  return compute_light(color, p.normal, p.reflection, intersection, true, depth, distance_from_eye + p.min_dist);
}

optional<Point> trace(const Vector& norm_ray, const Vector& origin, int depth, float distance_from_eye) {
  optional<Hit> tracer;
  for (const Ball& b : balls) {
    optional<Hit> t = b.pretrace(norm_ray, origin);
    if (tracer == nullopt || (t && t->closest_point_distance_from_viewer_ < tracer->closest_point_distance_from_viewer_)) {
      tracer = t;
    }
  }
  optional<Hit> t = pretrace_light(norm_ray, origin);
  if (t && (tracer == nullopt || t->closest_point_distance_from_viewer_ < tracer->closest_point_distance_from_viewer_)) {
    return light_trace(*t, norm_ray, origin, depth, distance_from_eye);
  }
  if (tracer != nullopt) {
    return ball_trace(*tracer, norm_ray, origin, depth, distance_from_eye);
  }
  return trace_room(norm_ray, origin, depth, distance_from_eye);
}

optional<float> distance(const Vector& norm_ray, const Vector& origin) {
  optional<Hit> ball_hit;
  for (const Ball& b : balls) {
    optional<Hit> another_ball_hit = b.pretrace(norm_ray, origin);
    if (ball_hit == nullopt ||
        (another_ball_hit && another_ball_hit->closest_point_distance_from_viewer_
           < ball_hit->closest_point_distance_from_viewer_)) {
      P("ball hit");
      ball_hit = another_ball_hit;
    }
  }
  optional<Hit> light_hit = pretrace_light(norm_ray, origin);
  if (light_hit && (ball_hit == nullopt
        || light_hit->closest_point_distance_from_viewer_ <
              ball_hit->closest_point_distance_from_viewer_)) {
    P("light hit");
    return light_hit->closest_point_distance_from_viewer_
      - sqrtf(light_size2 - light_hit->distance_from_object_center2_);
  }

  if (ball_hit != nullopt) {
    float distance_from_origin = ball_hit->closest_point_distance_from_viewer_
      - sqrtf(ball_size2 - ball_hit->distance_from_object_center2_);
    P("ball hit selected");
    return distance_from_origin;
  }

  RoomHit rt = pretrace_room(norm_ray, origin);
  P(rt.min_dist);
  return rt.min_dist;
}

Point compute_light(
    const Point& color,
    const Point& normal,
    const Point& reflection_in,
    const Point& point,
    bool rought_surface,
    int depth,
    float distance_from_eye) {
  Vector total_color = black;
  if (depth > 0) {
    Vector reflection = reflection_in;
    if (rought_surface) {
      reflection = (reflection + distr()).normalize();
    }
    auto second_ray = trace(reflection, point, depth - 1, distance_from_eye);
    if (second_ray) {
      total_color = (color * *second_ray) * defuse_attenuation;
    }
  }
  if (!rought_surface) {
    return total_color;
  }

  Vector light_rnd_pos = light_pos + light_distr();
  Vector light_from_point = light_rnd_pos - point;
  float angle_x_distance = dot(normal, light_from_point);
  if (angle_x_distance < 0) {
    return total_color;
  }
  float light_distance2 = light_from_point.size2();
  float light_distance_inv = 1/sqrtf(light_distance2);
  float light_distance = 1/light_distance_inv;
  Vector light_from_point_norm = light_from_point * light_distance_inv;

  for (auto b : balls) {
    auto res = b.pretrace(light_from_point_norm, point);
    if (res != nullopt) {
      if (res->closest_point_distance_from_viewer_< light_distance) {
        // Obstracted
        return total_color;
      }
    }
  }

  float angle = angle_x_distance * light_distance_inv;
  float total_distance = light_distance + distance_from_eye;
  Vector defuse_color = (color * light_color) * (angle / (total_distance * total_distance) * defuse_attenuation);
  total_color += defuse_color;
  return total_color;
}


Vector sight = Vector(0., 1, -0.1).normalize();
Vector sight_x, sight_y;

// Sort objects by distance / obstraction possibility
Uint8 colorToInt(float c) {
  if (c > 1) return 255;
  return sqrtf(c) * 255;
}
Vector saturateColor(Point c) {
  float m = std::max(c.x, c.y);
  if (m < 1) {
    return c;
  }
  float total = c.x + c.y + c.z;
  if (total > 3) {
    return Vector(1,1,1);
  }
  float scale = (3 - total) / (3 * m - total);
  float grey = 1 - scale * m;
  return Vector(grey + scale * c.x,
                grey + scale * c.y,
                grey + scale * c.z);
}


void check_saturation() {
  auto s = saturateColor(Vector(2, 0.1, 0));
  P(s);
  assert(s.sum() == Vector(1, 0.6, 0.5).sum());
}

Point viewer = Point(0, -5.5, 1.5);

std::vector<std::thread> threads;
std::mutex m;
std::condition_variable cv;
int frame = 0;
int base_frame = 0;
bool die = false;
bool accumulate = true;
int num_running = 0;
BasePoint<double>* fppixels;
Uint8* pixels;

void set_focus_distance(float x, float y) {
  Vector yoffset = sight_y * (float)(window_height / 2);
  Vector xoffset = sight_x * (float)(window_width / 2);

  Vector ray = sight - yoffset - xoffset + sight_y * y + sight_x * x;
  P(ray);
  Vector norm_ray = ray.normalize();
  P(norm_ray);
  auto res = distance(norm_ray, viewer);
  if (res) {
    focused_distance = *res;
  }
}

void drawThread(int id) {
  Vector yoffset = sight_y * (float)(window_height / 2);
  Vector xoffset = sight_x * (float)(window_width / 2);
  Uint8* my_pixels = pixels;
  BasePoint<double>* my_fppixels = fppixels;
  float num_frames = frame - base_frame;
  double base_mul = (num_frames - 1) / num_frames;
  double one_mul = 1 / num_frames;

  Vector yray = sight - yoffset - xoffset;
  for (int y = 0; y < window_height; y++) {
    if (y % numCPU == id) {
      Vector ray = yray;
      for (int x = 0; x < window_width; x++) {
        Vector norm_ray = ray;
        Vector focused_point = viewer + norm_ray * focused_distance;
        Vector me = viewer + sight_x.normalize() * (float)lense_gen(gen) + sight_y.normalize() * (float)lense_gen(gen);
        Vector new_ray = (focused_point - me).normalize();

        trace_values = x == 500 && y == 500;
        auto res = trace(new_ray, me, max_depth, 0);
        if (accumulate) {
          *my_fppixels = BasePoint<double>::convert(*my_fppixels) * base_mul + BasePoint<double>::convert(*res) * one_mul;
          res = BasePoint<float>::convert(*my_fppixels++);
        }
        if (res) {
          Vector saturated = saturateColor(*res);
          *my_pixels++ = colorToInt(saturated.x);
          *my_pixels++ = colorToInt(saturated.y);
          *my_pixels++ = colorToInt(saturated.z);
          *my_pixels++ = 255;
        }
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
    frame++;
    cv.notify_all();
  }
  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, []{return num_running == 0;});
  }
}
void start_accumulate() {
  accumulate = true;
  memset(fppixels, 0, window_width * window_height * sizeof(BasePoint<double>));
  base_frame = frame;
//  printf("start accumulate\n");
}

void update_viewpoint() {
  float dx = 1.9 / window_width;
  Vector x = cross(sight, Vector(0,0,1));
  sight_y = cross(sight, x);
  sight_x = cross(sight, sight_y);
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
    bool move_forward = false;
    bool move_backward = false;
    bool strafe_left = false;
    bool strafe_right = false;
    bool relative_motion = false;
    unsigned int currentTime = SDL_GetTicks();
    update_viewpoint();
    int mouse_x_before_rel = 0;
    int mouse_y_before_rel = 0;

    while (1) {
      unsigned int newTime = SDL_GetTicks();
      unsigned int dt = newTime - currentTime;

      bool moved = false;

      if (move_forward || move_backward) {
        Vector move = sight;
        move.z = 0;
        viewer += move * (0.001f * dt * (move_forward ? 1 : -1));
        //balls.back().position_ = viewer;
        //balls.back().position_.z = ball_size;
        moved = true;
      }

      if (strafe_left || strafe_right) {
        trace_values = true;
        Vector move = sight_x.normalize();
        move.z = 0;
        viewer += move * (0.001f * dt * (strafe_right ? 1 : -1));
        update_viewpoint();
        moved = true;
      }
      if (moved) start_accumulate();

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
                           start_accumulate();
                         }
                         break;
          case SDL_MOUSEMOTION:
                         if (event.motion.state == SDL_BUTTON_RMASK) {
                           Vector x = cross(sight, Vector(0.f,0.f,1.f)).normalize();
                           Vector y = cross(sight, x).normalize();
                           sight = (cross(sight, y) * (0.001f * event.motion.xrel) + sight).normalize();
                           sight = (cross(sight, x) * (0.001f * event.motion.yrel) + sight).normalize();
                           update_viewpoint();
                           start_accumulate();
                         }
                         break;
          case SDL_KEYUP:
          case SDL_KEYDOWN:
                         switch (event.key.keysym.scancode) {
                           case SDL_SCANCODE_ESCAPE:
                             die = true;
                             draw();
                             for (auto& t : threads) t.join();
                             goto exit;
                           case SDL_SCANCODE_W:
                             move_forward = event.key.state == SDL_PRESSED;
                             break;
                           case SDL_SCANCODE_S:
                             move_backward = event.key.state == SDL_PRESSED;
                             break;
                           case SDL_SCANCODE_A:
                             strafe_left = event.key.state == SDL_PRESSED;
                             break;
                           case SDL_SCANCODE_E:
                           case SDL_SCANCODE_D:
                             strafe_right = event.key.state == SDL_PRESSED;
                             break;
                           case SDL_SCANCODE_1:
                             max_depth = 0;
                             if (event.key.state != SDL_PRESSED) start_accumulate();
                             break;
                           case SDL_SCANCODE_2:
                             max_depth = 1;
                             if (event.key.state != SDL_PRESSED) start_accumulate();
                             break;
                           case SDL_SCANCODE_3:
                             max_depth = 2;
                             if (event.key.state != SDL_PRESSED) start_accumulate();
                             break;
                           case SDL_SCANCODE_4:
                             max_depth = 3;
                             if (event.key.state != SDL_PRESSED) start_accumulate();
                             break;
                           case SDL_SCANCODE_7:
                             wall_gen = std::normal_distribution<float>{-0.00,0.00};
                             if (event.key.state != SDL_PRESSED) start_accumulate();
                             break;
                           case SDL_SCANCODE_8:
                             wall_gen = std::normal_distribution<float>{-0.03,0.03};
                             if (event.key.state != SDL_PRESSED) start_accumulate();
                             break;
                           case SDL_SCANCODE_9:
                             wall_gen = std::normal_distribution<float>{-0.8,0.8};
                             if (event.key.state != SDL_PRESSED) start_accumulate();
                             break;
                           case SDL_SCANCODE_MINUS:
                             printf("Light size 0.1\n");
                             light_size = 0.1;
                             light_size2 = light_size * light_size;
                             light_inv_size = 1 / light_size;
                             light_gen = std::normal_distribution<float>{light_size, light_size};
                             if (event.key.state != SDL_PRESSED) start_accumulate();
                             break;
                           case SDL_SCANCODE_EQUALS:
                             printf("Light size 2\n");
                             light_size = 0.9;
                             light_size2 = light_size * light_size;
                             light_inv_size = 1 / light_size;
                             light_gen = std::normal_distribution<float>{light_size, light_size};
                             if (event.key.state != SDL_PRESSED) start_accumulate();
                             break;
                           case SDL_SCANCODE_LEFTBRACKET:
                             lense_blur = std::max(0.f, lense_blur * 0.8f - .0001f);
                             if (lense_blur == 0) {
                               printf("No blur\n");
                             }
                             lense_gen = std::normal_distribution<float>{-lense_blur,lense_blur};
                             if (event.key.state != SDL_PRESSED) start_accumulate();
                             break;

                           case SDL_SCANCODE_RIGHTBRACKET:
                             lense_blur = lense_blur * 1.2f + .0001f;
                             lense_gen = std::normal_distribution<float>{-lense_blur,lense_blur};
                             if (event.key.state != SDL_PRESSED) start_accumulate();
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
                               start_accumulate();
                             }
                             break;
                           case SDL_SCANCODE_Z:
                             if (event.key.state != SDL_PRESSED)
                               SDL_SetWindowGrab(window, SDL_FALSE);
                         }
        }
      }

      draw();
      SDL_UpdateTexture(texture, NULL, (void*)pixels, window_width * sizeof(Uint32));
      SDL_RenderCopy(renderer, texture, NULL, NULL);
      SDL_RenderPresent(renderer);
      event.type = -1;
      currentTime = newTime;
    }
exit:
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return EXIT_SUCCESS;
}
