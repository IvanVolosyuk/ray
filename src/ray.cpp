#include <stdio.h>
#include <vector>
#include <thread>
#include <mutex>
#include <stdlib.h>
#include <condition_variable>
#include <iostream>

#include <algorithm>
#include <random>
#include <functional>
#include <unistd.h>

#include "ray.hpp"
#include "gl_renderer.hpp"
#include "vector.hpp"

using namespace std::placeholders;

//#define P(x) print(#x, x);
#define P(x) {}

bool trace_values = false;
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

#include "shader/shader.h"

int window_width = 480;
int window_height = 270;
int max_depth = 2;

float distance(const vec3 norm_ray, const vec3 origin) {
  Hit hit = no_hit;
  for (size_t i = 0; i < LENGTH(balls); i++) {
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

std::vector<std::thread> threads;
std::mutex m;
std::condition_variable cv;
int frame = 0;
int base_frame = 0;
bool die = false;
int num_running = 0;
BasePoint<double>* fppixels;
Uint8* pixels;
int numCPU = 0;

void set_focus_distance(float x, float y) {
  vec3 yoffset = sight_y * (float)(window_height / 2);
  vec3 xoffset = sight_x * (float)(window_width / 2);

  vec3 ray = sight - yoffset - xoffset + sight_y * y + sight_x * x;
  P(ray);
  vec3 norm_ray = normalize(ray);
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
        vec3 focused_ray = normalize(ray + sight_x * antialiasing(gen) + sight_y * antialiasing(gen));
        vec3 focused_point = viewer + focused_ray * focused_distance;
        vec3 me = viewer + normalize(sight_x) * (float)lense_gen(gen) + normalize(sight_y) * (float)lense_gen(gen);
        vec3 new_ray = normalize(focused_point - me);

        trace_values = x == 500 && y == 500;
        auto res = (max_depth == 0) ? trace_0(new_ray, me, 0) 
          : ((max_depth == 1) ? trace_1(new_ray, me, 0)
              : (max_depth == 2) ? trace_2(new_ray, me, 0)
              : trace_3(new_ray, me, 0));
        // accumulate
        *my_fppixels += BasePoint<double>::convert(res);
        res = BasePoint<float>::convert(*my_fppixels++ * one_mul);

        vec3 saturated = saturateColor(res);
        *my_pixels++ = colorToInt(saturated.z);
        *my_pixels++ = colorToInt(saturated.y);
        *my_pixels++ = colorToInt(saturated.x);
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
  sight_x = normalize(cross(sight, vec3(0,0,1)));
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
    SDL_Renderer *renderer = nullptr;
    SDL_Window *window = nullptr;

    if (SDL_Init( SDL_INIT_VIDEO) < 0) {
      std::cerr << "Init failed: " << SDL_GetError() << std::endl;
      return 1;
    }


//    auto gl_renderer = OpenglRenderer::Create(window_width, window_height);
//    assert(gl_renderer.get() != nullptr);
//    SDL_Texture * texture = nullptr;


#if 1
    // FIXME: fallback
    SDL_CreateWindowAndRenderer(window_width, window_height, 0, &window, &renderer);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
    SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
    SDL_Texture * texture = SDL_CreateTexture(renderer,
                SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC, window_width, window_height);
    fppixels = new BasePoint<double>[window_width * window_height];
    pixels = new Uint8[window_width * window_height * 4];

    SDL_RenderPresent(renderer);
#endif
    Uint32 ts_move_forward;
    Uint32 ts_move_backward;
    Uint32 ts_strafe_left;
    Uint32 ts_strafe_right;

    bool relative_motion = false;
    update_viewpoint();

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
      auto move_left = std::bind(apply_motion, normalize(-sight_x), &ts_strafe_left, _1);
      auto move_right = std::bind(apply_motion, normalize(sight_x), &ts_strafe_right, _1);

      while(SDL_PollEvent(&event)) {
        switch (event.type) {
          case SDL_QUIT:
            goto exit;
            break;
          case SDL_MOUSEBUTTONUP:
                         if (event.button.button == SDL_BUTTON_RIGHT) {
                           SDL_SetRelativeMouseMode(SDL_FALSE);
                           relative_motion = false;
                         }
                         break;
          case SDL_MOUSEBUTTONDOWN:
                         if (event.button.button == SDL_BUTTON_RIGHT) {
                           SDL_SetRelativeMouseMode(SDL_TRUE);
                           SDL_SetWindowGrab(window, SDL_TRUE);
                           relative_motion = true;
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
                           vec3 x = normalize(cross(sight, vec3(0.f,0.f,1.f)));
                           vec3 y = normalize(cross(sight, x));
                           sight = normalize(cross(sight, y) * (-0.001f * event.motion.xrel) + sight);
                           sight = normalize(cross(sight, x) * (0.001f * event.motion.yrel) + sight);
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
                             if (false) {
                               die = true;
                               draw();
                               for (auto& t : threads) t.join();
                             }
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
                             // FIXME:
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
                             wall_gen = std::normal_distribution<float>{0.0,0.00};
                             if (event.key.state != SDL_PRESSED) reset_accumulate();
                             break;
                           case SDL_SCANCODE_8:
                             wall_gen = std::normal_distribution<float>{0.0,0.20};
                             if (event.key.state != SDL_PRESSED) reset_accumulate();
                             break;
                           case SDL_SCANCODE_9:
                             wall_gen = std::normal_distribution<float>{0.,0.4};
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
                             light_gen = std::normal_distribution<float>{0, light_size};
                             if (event.key.state != SDL_PRESSED) reset_accumulate();
                             break;
                           case SDL_SCANCODE_LEFTBRACKET:
                             lense_blur = std::max(0.f, lense_blur * 0.8f - .0001f);
                             if (lense_blur == 0) {
                               printf("No blur\n");
                             }
                             lense_gen = std::normal_distribution<float>{0,lense_blur};
                             if (event.key.state != SDL_PRESSED) reset_accumulate();
                             break;

                           case SDL_SCANCODE_RIGHTBRACKET:
                             lense_blur = lense_blur * 1.2f + .0001f;
                             lense_gen = std::normal_distribution<float>{0,lense_blur};
                             if (event.key.state != SDL_PRESSED) reset_accumulate();
                             break;
                           case SDL_SCANCODE_O:
                             diffuse_attenuation = 0.5;
                             if (event.key.state != SDL_PRESSED) reset_accumulate();
                             break;

                           case SDL_SCANCODE_P:
                             diffuse_attenuation = 0.9;
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
                           default:
                             // ignore
                             {}
                         }
        }
      }

      Uint32 newTime = SDL_GetTicks();
      move_forward(newTime);
      move_backward(newTime);
      move_left(newTime);
      move_right(newTime);

#if 1
        draw();
        SDL_UpdateTexture(texture, NULL, (void*)pixels, window_width * sizeof(Uint32));
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
#endif
//      gl_renderer->draw(viewer, sight, focused_distance);
    }
exit:
    if (false) {
      SDL_DestroyRenderer(renderer);
    }
    SDL_DestroyWindow(window);
    SDL_Quit();
    return EXIT_SUCCESS;

    }
