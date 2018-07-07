#include <stdio.h>

#include "ray.hpp"
#include "vector.hpp"

#include <SDL2/SDL.h>
#include <boost/optional.hpp>
#include <algorithm>

#define WINDOW_WIDTH 600
#define WINDOW_HEIGHT 600

#define P(x) print(#x, x);
// #define P(x) {}


using Vector = Point;
#define optional boost::optional
#define nullopt boost::none

Point viewer = Point(0, -6, 0.8);

Point light_pos = Point(-1.2, 0, 1.5);
float light_size = 1.0;
float light_power = 1.4;
Point light_color = Point(light_power, light_power, light_power);
float light_size2 = light_size * light_size;

float ball_size = 0.1;
float ball_size2 = ball_size * ball_size;
float ball_inv_size = 1 / ball_size;
bool trace_values = false;
Point ball_position = Point(0, 0, ball_size);
Point ball_color = Point(1, 0.01, 0.01);

Vector ball_vector = ball_position - viewer;
float ball_distance = ball_vector.size();
float ball_distance2 = ball_vector.size2();

Vector floor_color(0.01, 1.0, 0.01);
Vector floor_normal(0, 0, 1);



void print(const char* msg, const Vector& v) {
  if (trace_values)
    printf("%s: %f %f %f\n", msg, v.x, v.y, v.z);
}

void print(const char* msg, float v) {
  if (trace_values)
    printf("%s: %f\n", msg, v);
}

Point black = Point(0, 0, 0);

optional<Point> trace_extra(const Vector& norm_ray, const Vector& point);

Point ball_light(
    const Point& color,
    const Vector& normal,
    const Vector& reflection,
    const Point& point) {
  Vector reflection_color = color * 0.04;
  auto second_ray = trace_extra(reflection, point);
  if (second_ray) {
    reflection_color = *second_ray * 0.3 + color * ((*second_ray).size() * 0.1);
  }

  // TODO(vol): light is one point now
  Vector light_from_point = light_pos - point;
  float angle = light_from_point * normal;
  P(angle);
  if (angle < 0) {
    // No difuse color
    return reflection_color;
  }

  float light_distance2 = light_from_point.size2();
  P(light_distance2);
  Vector light_intensity = light_color * (angle/light_distance2);
  Point defuse_color = color.mul(light_intensity);
  return defuse_color + reflection_color;
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

float obstracted_by_ball(const Vector& origin, const Vector& norm_ray, float light_distance2) {
  P(norm_ray);
  Vector ball_vector = ball_position - origin;
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

Point floor_light(
    const Point& color,
    const Point& normal,
    const Point& point) {
  Vector light_from_floor = light_pos - point;
  float light_distance2 = light_from_floor.size2();
  P(light_distance2);
  Vector light_from_floor_norm = light_from_floor * (1/sqrt(light_distance2));
  float obstraction_level = obstracted_by_ball(point, light_from_floor_norm, light_distance2);
  if (obstraction_level > 0.99) {
    return black;
  }
  Vector light_intensity = light_color * ((1.-obstraction_level) * (normal * light_from_floor_norm)/light_distance2);
  Point defuse_color = color.mul(light_intensity);
  P(defuse_color);
  return defuse_color;
}

optional<Point> trace_ball(const Vector& norm_ray) {
  P(norm_ray);

  float closest_point_distance_from_viewer = norm_ray * ball_vector;
  if (closest_point_distance_from_viewer < 0) {
    return nullopt;
  }
  P(closest_point_distance_from_viewer);
  float distance_from_ball_center2 = ball_distance2 -
    closest_point_distance_from_viewer * closest_point_distance_from_viewer;
  if (distance_from_ball_center2 > ball_size2) {
    return nullopt;
  }
  float distance_from_viewer = closest_point_distance_from_viewer - sqrt(ball_size2 - distance_from_ball_center2);
  Point intersection = viewer + norm_ray * distance_from_viewer;

  P(intersection);
  Vector distance_from_ball_vector = intersection - ball_position;

  Vector normal = distance_from_ball_vector * ball_inv_size;
  P(normal);
  Vector ray_reflection = norm_ray - normal * 2 * (norm_ray * normal);
  P(ray_reflection);
  return ball_light(ball_color,
               normal,
               ray_reflection,
               intersection);
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

optional<Point> trace_floor(const Vector& norm_ray, const Vector& point) {
  P(norm_ray.z);
  if (norm_ray.z >= 0) return nullopt;
  Point ray = norm_ray * (-point.z / norm_ray.z);
  Point intersection = point + ray;
  P(intersection);
  return floor_light(floor_color, floor_normal, intersection);// * (1/(ray * ray));
}

optional<Point> trace_extra(const Vector& norm_ray, const Vector& point) {
  auto floor = trace_floor(norm_ray, point);
  if (floor) {
    return floor;
  }
  return trace_light(norm_ray, point);
}


optional<Point> trace(const Vector& norm_ray) {
  auto b = trace_ball(norm_ray);
  if (b) {
    return b;
  }
  return trace_extra(norm_ray, viewer);
}



Vector sight = Vector(0., 1, -0.1).normalize();

float dx = 0.1 / WINDOW_WIDTH;

Vector sight_x = Vector(dx, 0, 0);
Vector sight_y = (sight ^ sight_x).normalize() * dx;

// Sort objects by distance / obstraction possibility
Uint8 color(float c) {
  if (c > 1) return 255;
  return pow(c, 0.5f) * 255;
}

int main(void) {
    check_obstraction();
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

    Vector yray = sight - yoffset - xoffset;
    for (int y = 0; y < WINDOW_HEIGHT; y++) {
      Vector ray = yray;
      for (int x = 0; x < WINDOW_WIDTH; x++) {
	Vector norm_ray = ray.normalize();
        trace_values = x == 500 && y == 500;
	auto res = trace(norm_ray);
	if (res) {
	  SDL_SetRenderDrawColor(renderer, color(res->x), color(res->y), color(res->z), 255);
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
