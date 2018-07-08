#include <stdio.h>
#include <vector>
#include <thread>
#include <mutex>
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
float light_size = 2.0;
float light_power = 3.4;
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
float ray_attenuation = 0.6;
float defuse_attenuation = 0.4;

std::vector<int> max_rays = {1, 1, 5, 1, 1, 1};
int max_depth = 2;

std::random_device rd;
std::mt19937 gen(rd());
std::normal_distribution<> distr{-0.02,0.02};

int numCPU = sysconf(_SC_NPROCESSORS_ONLN);


void print(const char* msg, const Vector& v) {
  if (trace_values)
    printf("%s: %f %f %f\n", msg, v.x, v.y, v.z);
}

void print(const char* msg, float v) {
  if (trace_values)
    printf("%s: %f\n", msg, v);
}

Point black = Point(0, 0, 0);

optional<Point> trace_extra(const Vector& norm_ray, const Vector& point, int depth);
optional<Point> trace(const Vector& norm_ray, const Vector& origin, int depth);
Point floor_light(const Point& color, const Point& normal, const Point& point, const Point& reflection, int depth);
Point ball_light(const Point& color, const Vector& normal, const Vector& reflection, const Point& point, int depth);

float make_obstraction(float lsz, float sz, float off) {
  if (off > lsz + sz) {
    P(0);
    return 0;
  }
  if (off < sz - lsz) {
    return 1;
  }
  float max_possible = sz / lsz;
  P(max_possible);
  float fraction = -(off - lsz - sz) / (2 * lsz);
  P(fraction);
  auto res = std::max(0.f, std::min(1.f, max_possible * fraction));
  P(res);
  return res;
}

float check_obstraction() {
  trace_values = true;
  assert(make_obstraction(0.001, 0.1, 1) == 0);
  assert(make_obstraction(0.001, 0.1, 0) == 1);
  assert(make_obstraction(1, 1, 1) > 0.49f);
  assert(make_obstraction(1, 1, 1) < 0.51f);
}


class Ball;

struct Pretrace {
  const Ball *ball;
  float closest_point_distance_from_viewer;
  float distance_from_ball_center2;
};

class Ball {
  public:
  Ball(const Vector& position, const Vector& color) : position_(position), color_(color) {}

  optional<Pretrace> pretrace(const Vector& norm_ray, const Vector& origin) const {
    P(norm_ray);
    Vector ball_vector = position_ - origin;

    float closest_point_distance_from_viewer = norm_ray * ball_vector;
    if (closest_point_distance_from_viewer < 0) {
      return nullopt;
    }

    float ball_distance2 = ball_vector * ball_vector;
    float distance_from_ball_center2 = ball_distance2 -
      closest_point_distance_from_viewer * closest_point_distance_from_viewer;
    if (distance_from_ball_center2 > ball_size2) {
      return nullopt;
    }
    return Pretrace{this, closest_point_distance_from_viewer, distance_from_ball_center2};
  }

  Point trace(const Pretrace& p, const Vector& norm_ray, const Vector& origin, int depth) const {
    float distance_from_viewer = p.closest_point_distance_from_viewer - sqrt(ball_size2 - p.distance_from_ball_center2);
    Point intersection = origin + norm_ray * distance_from_viewer;

    P(intersection);
    Vector distance_from_ball_vector = intersection - position_;

    Vector normal = distance_from_ball_vector * ball_inv_size;
    P(normal);
    Vector ray_reflection = norm_ray - normal * 2 * (norm_ray * normal);
    P(ray_reflection);
    return ball_light(color_,
        normal,
        ray_reflection,
        intersection,
        depth);
  }

  float obstracted_by_ball(const Vector& origin, const Vector& norm_ray, float light_distance2) const {
    P(norm_ray);
    Vector ball_vector = position_ - origin;
    P(ball_vector);
    P(light_distance2);
    float ball_distance2 = ball_vector * ball_vector;
    P(ball_distance2);

    float closest_point_distance_from_viewer = norm_ray * ball_vector;
    if (closest_point_distance_from_viewer < 0) {
      return 0;
    }
    P(closest_point_distance_from_viewer);
    float distance_from_ball_center2 = ball_distance2 -
      closest_point_distance_from_viewer * closest_point_distance_from_viewer;

    // effective light size
    float lsz = sqrt(light_size2 / light_distance2);
    P(lsz);
    // effective obstruction size
    float sz = ball_size / closest_point_distance_from_viewer;
    P(sz);
    // effective shift of obstruction
    float off = sqrt(distance_from_ball_center2) / closest_point_distance_from_viewer;
    P(off);
    return make_obstraction(lsz, sz, off);
  }
  Vector position_, color_;
};

std::vector<Ball> balls = {
  {{0, 0.7 * ball_size, ball_size}, {1, 0.01, 0.01}},
  {{-2 * ball_size, 0, ball_size}, {0.01, 1.0, 0.01}},
  {{2 * ball_size, 0, ball_size}, {1.00, 0.00, 1.}},
  {{2 * ball_size, 0, ball_size}, {1.00, 1.00, 1.}},
};

optional<Point> trace_ball(const Vector& norm_ray, const Vector& origin, int depth) {
  optional<Pretrace> p;
  for (const Ball& b : balls) {
    optional<Pretrace> np = b.pretrace(norm_ray, origin);
    if (p == nullopt || (np && (*np).closest_point_distance_from_viewer < (*p).closest_point_distance_from_viewer)) {
        p = np;
    }
  }
  if (p == nullopt) return nullopt;
  return (*p).ball->trace(*p, norm_ray, origin, depth);
}

optional<Point> trace_light(const Vector& norm_ray, const Vector& origin) {
  P(norm_ray);
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

  float multiplier = ((light_size2 - distance_from_light_center2) / light_size2);
  multiplier += std::max(0.f, ((light_size2/4 - distance_from_light_center2) / light_size2) * 5);
  return light_color * (multiplier/light_distance2);
}

Point ball_light(
    const Point& color,
    const Vector& normal,
    const Vector& reflection,
    const Point& point,
    int depth) {
  Vector full_color = color * 0.0;
  if (depth > 0) {
    auto second_ray = trace(reflection, point, depth - 1);
    if (second_ray) {
      full_color += color.mul(*second_ray) * ray_attenuation;
    }
  }

  // TODO(vol): light is one point now
  Vector light_from_point = light_pos - point;
  float angle = light_from_point * normal;
  P(angle);
  if (angle < 0) {
    // No difuse color
    return full_color;
  }

  float light_distance2 = light_from_point.size2();
  P(light_distance2);
  Vector light_from_point_norm = light_from_point * (1/sqrt(light_distance2));

  float visibility_level = 1;
  for (const Ball& b : balls) {
    visibility_level *= 1 - b.obstracted_by_ball(point, light_from_point_norm, light_distance2);
  }
  if (visibility_level < 0.01) {
    return full_color;
  }

  Vector defuse_color = color.mul(light_color) * (visibility_level * angle * defuse_attenuation/light_distance2);
  full_color += defuse_color;
  return full_color;
}

Point floor_light(
    const Point& color,
    const Point& normal,
    const Point& point,
    const Point& reflection,
    int depth) {
  Vector total_color = black;
  if (depth > 0) {
    for (int i = 0; i < max_rays[depth]; i++) {
      Vector rand_reflection = Vector(
          reflection.x + distr(gen),
          reflection.y + distr(gen),
          reflection.z + distr(gen)).normalize();

      auto res = trace(rand_reflection, point, depth - 1);
      if (res) {
        total_color += color.mul(*res) * ray_attenuation;
      }
    }
    total_color *= (1. / max_rays[depth]);
  }
  Vector light_from_floor = light_pos - point;
  float light_distance2 = light_from_floor.size2();
  P(light_distance2);
  Vector light_from_floor_norm = light_from_floor * (1/sqrt(light_distance2));
  float visibility_level = 1;
  for (const Ball& b : balls) {
    visibility_level *= 1 - b.obstracted_by_ball(point, light_from_floor_norm, light_distance2);
  }
  if (visibility_level < 0.01) {
    return total_color;
  }
  Vector defuse_color = color.mul(light_color) * (visibility_level * defuse_attenuation * (normal * light_from_floor_norm)/light_distance2);
  P(defuse_color);
  total_color += defuse_color;
  return total_color;
}


float room_size = 6;
float ceiling_z = room_size;
float wall_x0 = -room_size;
float wall_x1 = room_size;
float wall_y0 = -room_size;
float wall_y1 = room_size;

optional<Point> trace_room(const Vector& norm_ray, const Vector& point, int depth) {
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
  return floor_light(color, normal, intersection, reflection, depth);
}

optional<Point> trace_floor(const Vector& norm_ray, const Vector& point, int depth) {

  P(norm_ray.z);
  if (norm_ray.z >= 0) return nullopt;
  Point ray = norm_ray * (-point.z / norm_ray.z);
  Point intersection = point + ray;
  P(intersection);
  Vector reflection = Vector(norm_ray.x, norm_ray.y, -norm_ray.z);
  return floor_light(floor_color, floor_normal, intersection, reflection, depth);
}

optional<Point> trace_extra(const Vector& norm_ray, const Vector& point, int depth) {
  auto floor = trace_room(norm_ray, point, depth);
  if (floor) {
    return floor;
  }
  return trace_light(norm_ray, point);
}


optional<Point> trace(const Vector& norm_ray, const Vector& origin, int depth) {
  auto b = trace_ball(norm_ray, origin, depth);
  if (b) {
    return b;
  }
  return trace_extra(norm_ray, origin, depth);
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
bool die = false;
int num_running = 0;
Uint8 *pixels;


void drawThread(int id) {
  Vector yoffset = sight_y * (WINDOW_HEIGHT / 2);
  Vector xoffset = sight_x * (WINDOW_WIDTH / 2);
  Uint8* my_pixels = pixels;

  Vector yray = sight - yoffset - xoffset;
  for (int y = 0; y < WINDOW_HEIGHT; y++) {
    if (y % numCPU == id) {
      Vector ray = yray;
      for (int x = 0; x < WINDOW_WIDTH; x++) {
        Vector norm_ray = ray.normalize();
        trace_values = x == 500 && y == 500;
        auto res = trace(norm_ray, viewer, max_depth);
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

int main(void) {
    check_obstraction();
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
      if (move_forward || move_backward) {
        Vector move = sight;
        move.z = 0;
        viewer += move * (0.001 * dt * (move_forward ? 1 : -1));
        balls[3].position_ = viewer;
        balls[3].position_.z = ball_size;
      }

      if (turn_left || turn_right) {
        trace_values = true;
        P(sight);
        sight = (((sight ^ Vector(0,0,turn_left ? -1 : 1)) * (0.001 * dt) + sight)).normalize();
        P(sight);
        P(sight_x);
        sight_x = sight ^ Vector(0,0,dx);
        P(sight_x);
        P(sight_y);
        sight_y = (sight ^ sight_x).normalize() * dx;
        P(sight_y);
      }

      while(SDL_PollEvent(&event)) {
        switch (event.type) {
          case SDL_QUIT:
            exit(0);
            break;
          case SDL_MOUSEBUTTONUP:
                         if (event.button.button == SDL_BUTTON_LEFT)
                           move_forward = false;
                         break;
          case SDL_MOUSEBUTTONDOWN:
                         if (event.button.button == SDL_BUTTON_LEFT)
                           move_forward = true;
                         break;
          case SDL_KEYDOWN:
          case SDL_KEYUP:
                         switch (event.key.keysym.scancode) {
                           case SDL_SCANCODE_ESCAPE:
                             die = true;
                             draw();
                             for (auto& t : threads) t.join();
                             exit(0);
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
                             max_depth = 1;
                             break;
                           case SDL_SCANCODE_2:
                             max_depth = 2;
                             break;
                           case SDL_SCANCODE_3:
                             max_depth = 3;
                             break;
                           case SDL_SCANCODE_5:
                             max_rays[0] = 1;
                             max_rays[1] = 1;
                             max_rays[2] = 1;
                             max_rays[3] = 1;
                             max_rays[3] = 1;
                             max_rays[max_depth] = 5;
                             break;
                           case SDL_SCANCODE_6:
                             max_rays[0] = 1;
                             max_rays[1] = 1;
                             max_rays[2] = 1;
                             max_rays[3] = 1;
                             max_rays[3] = 1;
                             break;
                           case SDL_SCANCODE_7:
                             distr = std::normal_distribution<double>{-0.00,0.00};
                             break;
                           case SDL_SCANCODE_8:
                             distr = std::normal_distribution<double>{-0.03,0.03};
                             break;
                           case SDL_SCANCODE_9:
                             distr = std::normal_distribution<double>{-0.3,0.3};
                             break;
                           case SDL_SCANCODE_MINUS:
                             printf("Light size 0.1\n");
                             light_size = 0.1;
                             light_size2 = light_size * light_size;
                             break;
                           case SDL_SCANCODE_EQUALS:
                             printf("Light size 2\n");
                             light_size = 2;
                             light_size2 = light_size * light_size;
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
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return EXIT_SUCCESS;
}
