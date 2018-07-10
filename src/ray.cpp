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

#define WINDOW_WIDTH 600
#define WINDOW_HEIGHT 600

//#define P(x) print(#x, x);
#define P(x) {}


using Vector = Point;
#define optional boost::optional
#define nullopt boost::none

Point light_pos = Point(-4.2, -3, 2);
float light_size = 0.9;
float light_power = 50.4;
Point light_color = Point(light_power, light_power, light_power);
float light_size2 = light_size * light_size;

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
std::normal_distribution<> wall_gen{-0.02,0.02};
std::normal_distribution<> light_gen{light_size, light_size};

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

class Tracer {
  public:
    const Ball *ball_;
    float closest_point_distance_from_viewer_;
    float distance_from_object_center2_;
};


class Ball {
  public:
    Ball(const Vector& position, const Vector& color) : position_(position), color_(color) {}

    optional<Tracer> pretrace(const Vector& norm_ray, const Vector& origin) const {
      P(norm_ray);
      Vector ball_vector = position_ - origin;

      float closest_point_distance_from_viewer = norm_ray * ball_vector;
      if (closest_point_distance_from_viewer < 0) {
        return nullopt;
      }

      float ball_distance2 = ball_vector * ball_vector;
      float distance_from_object_center2 = ball_distance2 -
        closest_point_distance_from_viewer * closest_point_distance_from_viewer;
      if (distance_from_object_center2 > ball_size2) {
        return nullopt;
      }
      return Tracer{this, closest_point_distance_from_viewer, distance_from_object_center2};
    }

    Vector position_, color_;
};

Point ball_trace(const Tracer& p, const Vector& norm_ray, const Vector& origin, int depth, float distance_from_eye) {
  float distance_from_origin = p.closest_point_distance_from_viewer_ - sqrt(ball_size2 - p.distance_from_object_center2_);
  Point intersection = origin + norm_ray * distance_from_origin;

  P(intersection);
  Vector distance_from_ball_vector = intersection - p.ball_->position_;

  Vector normal = distance_from_ball_vector * ball_inv_size;
  P(normal);
  Vector ray_reflection = norm_ray - normal * (2 * (norm_ray * normal));
  P(ray_reflection);
  return compute_light(p.ball_->color_,
      normal,
      ray_reflection,
      intersection,
      false,
      depth,
      distance_from_eye + distance_from_origin);
}

Point light_trace(const Tracer& p, int depth, float distance_from_eye) {
  return light_color * (1/(distance_from_eye * distance_from_eye));
}

class optional<Tracer> pretrace_light(const Vector& norm_ray, const Vector& origin) {
  Vector light_vector = light_pos - origin;
  float light_distance2 = light_vector * light_vector;

  float closest_point_distance_from_origin = norm_ray * light_vector;
  if (closest_point_distance_from_origin < 0) {
    return nullopt;
  }

  P(closest_point_distance_from_origin);
  float distance_from_light_center2 = light_distance2 -
    closest_point_distance_from_origin * closest_point_distance_from_origin;
  if (distance_from_light_center2 > light_size2) {
    return nullopt;
  }
  return Tracer{nullptr, closest_point_distance_from_origin, distance_from_light_center2};
}

std::vector<Ball> balls = {
  {{0, 0.7f * ball_size, ball_size}, {1, 0.01, 0.01}},
  {{-2 * ball_size, 0, ball_size}, {0.01, 1.0, 0.01}},
  {{2 * ball_size, 0, ball_size}, {1.00, 0.00, 1.}},
  {{2 * ball_size, 0, ball_size}, {1.00, 1.00, 1.}},
};

optional<Point> trace_objects(const Vector& norm_ray, const Vector& origin, int depth, float distance_from_eye) {
  optional<Tracer> tracer;
  for (const Ball& b : balls) {
    optional<Tracer> t = b.pretrace(norm_ray, origin);
    if (tracer == nullopt || (t && t->closest_point_distance_from_viewer_ < tracer->closest_point_distance_from_viewer_)) {
      tracer = t;
    }
  }
  optional<Tracer> t = pretrace_light(norm_ray, origin);
  if (t && (tracer == nullopt || t->closest_point_distance_from_viewer_ < tracer->closest_point_distance_from_viewer_)) {
    return light_trace(*t, depth, distance_from_eye);
  }
  if (tracer != nullopt) {
    return ball_trace(*tracer, norm_ray, origin, depth, distance_from_eye);
  }
  return nullopt;
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
      total_color = color.mul(*second_ray) * defuse_attenuation;
    }
  }
  Vector light_rnd_pos = light_pos + light_distr();
  Vector light_from_point = light_rnd_pos - point;
  float angle_x_distance = normal * light_from_point;
  if (angle_x_distance < 0) {
    return total_color;
  }
  float light_distance2 = light_from_point.size2();
  float light_distance_inv = 1/sqrt(light_distance2);
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
  Vector defuse_color = color.mul(light_color) * (angle / (total_distance * total_distance)) * defuse_attenuation;
  total_color += defuse_color;
  return total_color;
}


float room_size = 6;
float ceiling_z = room_size;
float wall_x0 = -room_size;
float wall_x1 = room_size;
float wall_y0 = -room_size;
float wall_y1 = room_size;

optional<Point> trace_room(const Vector& norm_ray, const Vector& point, int depth, float distance_from_eye) {
  float dist_x, dist_y, dist_z;
  if (norm_ray.z >= 0) {
    // trace ceiling
    dist_z = (ceiling_z-point.z) / norm_ray.z;
  } else {
    // trace floor
    dist_z = (/*0*/-point.z) / norm_ray.z;
  }
  if (norm_ray.x >= 0) {
    // trace ceiling
    dist_x = (wall_x1-point.x) / norm_ray.x;
  } else {
    // trace floor
    dist_x = (wall_x0-point.x) / norm_ray.x;
  }

  if (norm_ray.y >= 0) {
    // trace ceiling
    dist_y = (wall_y1-point.y) / norm_ray.y;
  } else {
    // trace floor
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

  Point ray = norm_ray * min_dist;
  Point intersection = point + ray;
  return compute_light(color, normal, reflection, intersection, true, depth, distance_from_eye + min_dist);
}

optional<Point> trace(const Vector& norm_ray, const Vector& origin, int depth, float distance_from_eye) {
  auto b = trace_objects(norm_ray, origin, depth, distance_from_eye);
  if (b) {
    return b;
  }
  return trace_room(norm_ray, origin, depth, distance_from_eye);
}

Vector sight = Vector(0., 1, -0.1).normalize();

float dx = 1.9 / WINDOW_WIDTH;

Vector sight_x = Vector(dx, 0, 0);
Vector sight_y = (sight ^ sight_x).normalize() * dx;

// Sort objects by distance / obstraction possibility
Uint8 colorToInt(float c) {
  if (c > 1) return 255;
  return sqrt(c) * 255;
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

Point viewer = Point(0, -6, 2.0);

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


void drawThread(int id) {
  Vector yoffset = sight_y * (float)(WINDOW_HEIGHT / 2);
  Vector xoffset = sight_x * (float)(WINDOW_WIDTH / 2);
  Uint8* my_pixels = pixels;
  BasePoint<double>* my_fppixels = fppixels;
  float num_frames = frame - base_frame;
  double base_mul = (num_frames - 1) / num_frames;
  double one_mul = 1 / num_frames;

  Vector yray = sight - yoffset - xoffset;
  for (int y = 0; y < WINDOW_HEIGHT; y++) {
    if (y % numCPU == id) {
      Vector ray = yray;
      for (int x = 0; x < WINDOW_WIDTH; x++) {
        Vector norm_ray = ray.normalize();
        trace_values = x == 500 && y == 500;
        auto res = trace(norm_ray, viewer, max_depth, 0);
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
      my_pixels += WINDOW_WIDTH * 4;
      my_fppixels += WINDOW_WIDTH;
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
  memset(fppixels, 0, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(BasePoint<double>));
  base_frame = frame;
//  printf("start accumulate\n");
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
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
    SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
    SDL_Texture * texture = SDL_CreateTexture(renderer,
                SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC, WINDOW_WIDTH, WINDOW_HEIGHT);
    fppixels = new BasePoint<double>[WINDOW_WIDTH * WINDOW_HEIGHT];
    pixels = new Uint8[WINDOW_WIDTH * WINDOW_HEIGHT * 4];

    SDL_RenderPresent(renderer);
    bool move_forward = false;
    bool move_backward = false;
    bool turn_left = false;
    bool turn_right = false;
    unsigned int currentTime = SDL_GetTicks();

    while (1) {
      unsigned int newTime = SDL_GetTicks();
      unsigned int dt = newTime - currentTime;

      bool moved = false;

      if (move_forward || move_backward) {
        Vector move = sight;
        move.z = 0;
        viewer += move * (0.001f * dt * (move_forward ? 1 : -1));
        balls.back().position_ = viewer;
        balls.back().position_.z = ball_size;
        moved = true;
      }

      if (turn_left || turn_right) {
        trace_values = true;
        P(sight);
        sight = (((sight ^ Vector(0.f,0.f,turn_left ? -1.f : 1.f)) * (0.001f * dt) + sight)).normalize();
        P(sight);
        P(sight_x);
        sight_x = sight ^ Vector(0.f,0.f,dx);
        P(sight_x);
        P(sight_y);
        sight_y = (sight ^ sight_x).normalize() * dx;
        P(sight_y);
        moved = true;
      }
      if (moved) start_accumulate();

      while(SDL_PollEvent(&event)) {
        switch (event.type) {
          case SDL_QUIT:
            goto exit;
            break;
          case SDL_MOUSEBUTTONUP:
                         if (event.button.button == SDL_BUTTON_LEFT)
                           move_forward = false;
                         break;
          case SDL_MOUSEBUTTONDOWN:
                         if (event.button.button == SDL_BUTTON_LEFT)
                           move_forward = true;
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
                             turn_left = event.key.state == SDL_PRESSED;
                             break;
                           case SDL_SCANCODE_E:
                           case SDL_SCANCODE_D:
                             turn_right = event.key.state == SDL_PRESSED;
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
                             wall_gen = std::normal_distribution<double>{-0.00,0.00};
                             if (event.key.state != SDL_PRESSED) start_accumulate();
                             break;
                           case SDL_SCANCODE_8:
                             wall_gen = std::normal_distribution<double>{-0.03,0.03};
                             if (event.key.state != SDL_PRESSED) start_accumulate();
                             break;
                           case SDL_SCANCODE_9:
                             wall_gen = std::normal_distribution<double>{-0.3,0.3};
                             if (event.key.state != SDL_PRESSED) start_accumulate();
                             break;
                           case SDL_SCANCODE_MINUS:
                             printf("Light size 0.1\n");
                             light_size = 0.1;
                             light_size2 = light_size * light_size;
                             light_gen = std::normal_distribution<double>{light_size, light_size};
                             if (event.key.state != SDL_PRESSED) start_accumulate();
                             break;
                           case SDL_SCANCODE_EQUALS:
                             printf("Light size 2\n");
                             light_size = 0.9;
                             light_size2 = light_size * light_size;
                             light_gen = std::normal_distribution<double>{light_size, light_size};
                             if (event.key.state != SDL_PRESSED) start_accumulate();
                             break;
                         }
        }
      }

      draw();
      SDL_UpdateTexture(texture, NULL, (void*)pixels, WINDOW_WIDTH * sizeof(Uint32));
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
