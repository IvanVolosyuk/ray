#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <iostream>

#include <algorithm>
#include <random>
#include <functional>

#include "ray.hpp"
#include "gl_renderer.hpp"
#include "sw_renderer.hpp"
#include "vector.hpp"
#include "shader/input.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_main.h>

using namespace std::placeholders;

int window_width = 480;
int window_height = 270;

void set_focus_distance(float x, float y) {
  focused_distance = SoftwareRenderer::distance(x, y, window_width, window_height);
}

int run_gl = false;

std::unique_ptr<Renderer> MakeRenderer() {
  std::unique_ptr<Renderer> r;
  if (run_gl) {
    r = OpenglRenderer::Create(window_width, window_height);
  }
  if (r.get() == nullptr) {
    r = SoftwareRenderer::Create(window_width, window_height);
  }
  assert(r.get() != nullptr);
  return r;
}

void update_viewpoint() {
//  float dx = 1.9 / window_width;
  sight_x = normalize(cross(sight, vec3(0,0,1)));
  sight_y = cross(sight, sight_x);
}

int main(int argc, char** argv) {
    SDL_Event event;

    if (SDL_Init( SDL_INIT_VIDEO) < 0) {
      std::cerr << "Init failed: " << SDL_GetError() << std::endl;
      return 1;
    }

    auto renderer = MakeRenderer();

    Uint32 ts_move_forward;
    Uint32 ts_move_backward;
    Uint32 ts_strafe_left;
    Uint32 ts_strafe_right;

    bool relative_motion = false;
    update_viewpoint();

    auto apply_motion = [&renderer](vec3 dir, Uint32* prev_ts, Uint32 ts) {
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
      renderer->reset_accumulate();
    };

    printf(R"(
Controls:

Escape : Exit

Left Mouse Button         : Focus point
Right Mouse Button + Drag : Rotate

W,S,A,E = Move
1,2,3,4 = Depth of ray tracing
7,8,9   = Change reflectivity of walls
-,+     = Light source size
O,P     = Diffuse attenuation
[,]     = Amount of depth of field effect
         (press multiple times)
F       = Switch to 1920x1080
G       = Switch Sofware / OpenGL renderer
)");

    while (1) {
      auto move_forward = std::bind(apply_motion, sight, &ts_move_forward, _1);
      auto move_backward = std::bind(apply_motion, -sight, &ts_move_backward, _1);
      auto move_left = std::bind(apply_motion, -sight_x, &ts_strafe_left, _1);
      auto move_right = std::bind(apply_motion, sight_x, &ts_strafe_right, _1);

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
                           SDL_SetWindowGrab(renderer->GetWindow(), SDL_TRUE);
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
                           renderer->reset_accumulate();
                         }
                         break;
          case SDL_MOUSEMOTION:
                         if (event.motion.state == SDL_BUTTON_RMASK) {
                           vec3 x = normalize(cross(sight, vec3(0.f,0.f,1.f)));
                           vec3 y = normalize(cross(sight, x));
                           sight = normalize(cross(sight, y) * (-0.0001f * event.motion.xrel) + sight);
                           sight = normalize(cross(sight, x) * (0.0001f * event.motion.yrel) + sight);
                           update_viewpoint();
                           renderer->reset_accumulate();
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
                             renderer.reset();
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
                             if (event.key.state != SDL_PRESSED) renderer->reset_accumulate();
                             break;
                           case SDL_SCANCODE_2:
                             max_depth = 1;
                             if (event.key.state != SDL_PRESSED) renderer->reset_accumulate();
                             break;
                           case SDL_SCANCODE_3:
                             max_depth = 2;
                             if (event.key.state != SDL_PRESSED) renderer->reset_accumulate();
                             break;
                           case SDL_SCANCODE_4:
                             max_depth = 3;
                             if (event.key.state != SDL_PRESSED) renderer->reset_accumulate();
                             break;
                           case SDL_SCANCODE_7:
                             if (event.key.state != SDL_PRESSED) renderer->reset_accumulate();
                             wall_distribution = 0.0;
                             break;
                           case SDL_SCANCODE_8:
                             wall_distribution = 0.20;
                             if (event.key.state != SDL_PRESSED) renderer->reset_accumulate();
                             break;
                           case SDL_SCANCODE_9:
                             wall_distribution = 0.4;
                             if (event.key.state != SDL_PRESSED) renderer->reset_accumulate();
                             break;
                           case SDL_SCANCODE_MINUS:
                             printf("Light size 0.1\n");
                             light_size = 0.1;
                             light_size2 = light_size * light_size;
                             light_inv_size = 1 / light_size;
                             if (event.key.state != SDL_PRESSED) renderer->reset_accumulate();
                             break;
                           case SDL_SCANCODE_EQUALS:
                             printf("Light size 2\n");
                             light_size = 0.9;
                             light_size2 = light_size * light_size;
                             light_inv_size = 1 / light_size;
                             if (event.key.state != SDL_PRESSED) renderer->reset_accumulate();
                             break;
                           case SDL_SCANCODE_LEFTBRACKET:
                             lense_blur = std::max(0.f, lense_blur * 0.8f - .0001f);
                             if (lense_blur == 0) {
                               printf("No blur\n");
                             }
                             if (event.key.state != SDL_PRESSED) renderer->reset_accumulate();
                             break;

                           case SDL_SCANCODE_RIGHTBRACKET:
                             lense_blur = lense_blur * 1.2f + .0001f;
                             if (event.key.state != SDL_PRESSED) renderer->reset_accumulate();
                             break;
                           case SDL_SCANCODE_O:
                             diffuse_attenuation = 0.5;
                             if (event.key.state != SDL_PRESSED) renderer->reset_accumulate();
                             break;

                           case SDL_SCANCODE_P:
                             diffuse_attenuation = 0.9;
                             if (event.key.state != SDL_PRESSED) renderer->reset_accumulate();
                             break;
                           case SDL_SCANCODE_G:
                             if (event.key.state != SDL_PRESSED) {
                               run_gl = !run_gl;
                               renderer.reset();
                               renderer = MakeRenderer();
                             }
                             break;
                           case SDL_SCANCODE_F:
                             if (event.key.state != SDL_PRESSED) {

                               if (window_width == 480) {
                                 window_width = 1920;
                                 window_height = 1080;
                               } else {
                                 window_width = 480;
                                 window_height = 270;
                               }
                               renderer.reset();
                               renderer = MakeRenderer();
                             }
                             break;
                           case SDL_SCANCODE_Z:
                             if (event.key.state != SDL_PRESSED)
                               SDL_SetWindowGrab(renderer->GetWindow(), SDL_FALSE);
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

//      printf("Sight_x: %f %f %f\n", sight_x.x, sight_x.y, sight_x.z);
//      printf("Sight_y: %f %f %f\n", sight_y.x, sight_y.y, sight_y.z);
      renderer->draw();
    }
exit:
    renderer.reset();
    SDL_Quit();
    return EXIT_SUCCESS;
}
