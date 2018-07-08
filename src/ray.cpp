#include <stdio.h>
#include <vector>

#include "ray.hpp"
#include "vector.hpp"

#include <SDL2/SDL.h>
#include <boost/optional.hpp>
#include <algorithm>
#include <random>

#define WINDOW_WIDTH 600
#define WINDOW_HEIGHT 600

#define P(x) print(#x, x);
// #define P(x) {}


using Vector = Point;
#define optional boost::optional
#define nullopt boost::none

Point light_pos = Point(-4.2, 0, 1);
float light_size = 1.0;
float light_power = 7.4;
Point light_color = Point(light_power, light_power, light_power);
float light_size2 = light_size * light_size;

float ball_size = 0.1;
float ball_size2 = ball_size * ball_size;
float ball_inv_size = 1 / ball_size;
bool trace_values = false;

Vector floor_color(0.14, 1.0, 0.14);
Vector wall_color(0.85, 0.8, 0.48);
Vector ceiling_color(0.88, 0.88, 0.88);
Vector floor_normal(0, 0, 1);
float attenuation = 0.6;

std::random_device rd;
std::mt19937 gen(rd());
std::normal_distribution<> distr{-0.02,0.02};


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
      full_color += color.mul(*second_ray) * attenuation;
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
  Vector defuse_color = color.mul(light_color) * (angle/light_distance2);
  full_color += defuse_color;
  return full_color;
}

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

  optional<Pretrace> pretrace(const Vector& norm_ray, const Vector& origin, int depth) const {
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
  {{0, 0, ball_size}, {1, 0.01, 0.01}},
  {{2 * ball_size, 2 * ball_size, ball_size}, {0.01, 1.0, 0.01}},
  {{-2 * ball_size, 2 * ball_size, ball_size}, {0.01, 0.01, 1.}},
};

optional<Point> trace_ball(const Vector& norm_ray, const Vector& origin, int depth) {
  optional<Pretrace> p;
  for (const Ball& b : balls) {
    optional<Pretrace> np = b.pretrace(norm_ray, origin, depth);
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

Point floor_light(
    const Point& color,
    const Point& normal,
    const Point& point,
    const Point& reflection,
    int depth) {
  Vector total_color = black;
  if (depth > 0) {
    Vector rand_reflection = Vector(
        reflection.x + distr(gen),
        reflection.y + distr(gen),
        reflection.z + distr(gen)).normalize();

    auto res = trace(rand_reflection, point, depth - 1);
    if (res) {
      total_color += color.mul(*res) * attenuation;
    }
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
  Vector defuse_color = color.mul(light_color) * (visibility_level * (normal * light_from_floor_norm)/light_distance2);
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

float dx = 0.1 / WINDOW_WIDTH;

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

    Vector yoffset = sight_y * (WINDOW_HEIGHT / 2);
    Vector xoffset = sight_x * (WINDOW_WIDTH / 2);
    Point viewer = Point(0, -6, 0.8);
    int max_depth = 4;

    Vector yray = sight - yoffset - xoffset;
    for (int y = 0; y < WINDOW_HEIGHT; y++) {
      Vector ray = yray;
      for (int x = 0; x < WINDOW_WIDTH; x++) {
	Vector norm_ray = ray.normalize();
        trace_values = x == 500 && y == 500;
	auto res = trace(norm_ray, viewer, max_depth);
	if (res) {
          Vector saturated = saturateColor(*res);
          SDL_SetRenderDrawColor(
              renderer,
              colorToInt(saturated.x),
              colorToInt(saturated.y),
              colorToInt(saturated.z), 255);
          SDL_RenderDrawPoint(renderer, x, y);
        }
	ray += sight_x;
      }
      yray += sight_y;
    }

    SDL_RenderPresent(renderer);
    while (1) {
        if (SDL_PollEvent(&event) && event.type == SDL_QUIT)
            break;
    }
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return EXIT_SUCCESS;
}
