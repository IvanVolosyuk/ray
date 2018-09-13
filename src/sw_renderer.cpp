#include "sw_renderer.hpp"
#include <unistd.h>
#include <iostream>

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

std::unique_ptr<Renderer> SoftwareRenderer::Create(int window_width, int window_height) {
  std::unique_ptr<SoftwareRenderer> r(new SoftwareRenderer());
  r->window_width_ = window_width;
  r->window_height_ = window_height;

  const char* cpus = getenv("NUM_CPUS");
  if (cpus != nullptr) {
    r->numCPU_ = atoi(cpus);
  }
  if (r->numCPU_ == 0) {
    r->numCPU_ = std::thread::hardware_concurrency();
  }
  printf("Software renderer initializing... CPUs: %d\n", r->numCPU_);
  fflush(stdout);

  SDL_CreateWindowAndRenderer(
      r->window_width_, r->window_height_, 0, &r->window_, &r->renderer_);
  SDL_SetRenderDrawColor(r->renderer_, 0, 0, 0, 0);
  SDL_RenderClear(r->renderer_);
  SDL_SetRenderDrawColor(r->renderer_, 255, 0, 0, 255);
  r->texture_ = SDL_CreateTexture(r->renderer_,
      SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC, window_width, window_height);
  r->fppixels_ = new BasePoint<double>[window_width * window_height];
  r->pixels_ = new Uint8[window_width * window_height * 4];
  r->pp_fppixels_ = new vec3[window_width * window_height];
  r->pp_fppixels2_ = new vec3[window_width * window_height];

  SDL_RenderPresent(r->renderer_);
  return r;
}

SoftwareRenderer::~SoftwareRenderer() {
  SDL_DestroyRenderer(renderer_);
  SDL_DestroyWindow(window_);
  SDL_DestroyTexture(texture_);

  die_ = true;
  draw();
  for (auto& t : threads_) t.join();

  delete fppixels_;
  delete pixels_;
  delete pp_fppixels_;
  delete pp_fppixels2_;
}

float SoftwareRenderer::distance(float x, float y, int window_width, int window_height) {
  vec3 xoffset = sight_x * fov;
  vec3 yoffset = sight_y * (fov * window_height / window_width);
  vec3 dx = xoffset * (1.f/(window_width / 2));
  vec3 dy = yoffset * (1.f/(window_height / 2));
  vec3 ray = (sight - yoffset - xoffset + dx * x + dy * y);
  vec3 norm_ray = normalize(ray);
  // Return distance to the plane instead
  float ray_len = ray.size();

  Hit hit = no_hit;
  for (size_t i = 0; i < LENGTH(balls); i++) {
    Hit another_hit = ball_hit(i, norm_ray, viewer);
    if (another_hit.closest_point_distance_from_viewer_
           < hit.closest_point_distance_from_viewer_) {
      hit = another_hit;
    }
  }
  Hit light = light_hit(norm_ray, viewer);
  if (light.closest_point_distance_from_viewer_ <
              hit.closest_point_distance_from_viewer_) {
    return (light.closest_point_distance_from_viewer_
      - sqrtf(light_size2 - light.distance_from_object_center2_)) / ray_len;
  }

  if (hit.id_ >= 0) {
    float distance_from_origin = hit.closest_point_distance_from_viewer_
      - sqrtf(ball_size2 - hit.distance_from_object_center2_);
    return distance_from_origin / ray_len;
  }

  RoomHit rt = room_hit(norm_ray, viewer);
  P(rt.min_dist);
  return rt.min_dist / ray_len;
}

void SoftwareRenderer::adjust(float x, float y, int window_width, int window_height, int mode) {
  vec3 xoffset = sight_x * fov;
  vec3 yoffset = sight_y * (fov * window_height / window_width);
  vec3 dx = xoffset * (1.f/(window_width / 2));
  vec3 dy = yoffset * (1.f/(window_height / 2));
  vec3 ray = (sight - yoffset - xoffset + dx * x + dy * y);
  vec3 norm_ray = normalize(ray);
  // Return distance to the plane instead

  Hit hit = no_hit;
  for (size_t i = 0; i < LENGTH(balls); i++) {
    Hit another_hit = ball_hit(i, norm_ray, viewer);
    if (another_hit.closest_point_distance_from_viewer_
           < hit.closest_point_distance_from_viewer_) {
      hit = another_hit;
    }
  }
  Hit light = light_hit(norm_ray, viewer);
  if (light.closest_point_distance_from_viewer_ <
              hit.closest_point_distance_from_viewer_) {
    return;
  }

  auto adj = [&](const char* name, float *exp, float *di) {
    switch (mode) {
      case 0: *exp = 32;
              break;
      case 1: *exp = (*exp == 0) ? 1 : (*exp * 2);
              break;
      case -1: *exp /= 2;
               break;
      case 9: *di = std::max(0.f, *di - 0.05f);
              break;
      case 10: *di = std::min(1.f, *di + 0.05f);
               break;
    }
    std::cerr << name << ": exp: " << *exp << " diffuse_ammount: " << *di << std::endl;
  };

  if (hit.id_ >= 0) {
    adj("Ball", &balls[hit.id_].material_.specular_exponent_,
        &balls[hit.id_].material_.diffuse_ammount_);
    return;
  }

  RoomHit rt = room_hit(norm_ray, viewer);
  if (rt.normal.z == 1) {
    adj("Floor", &floor_tex->specular_exponent_, &floor_tex->diffuse_ammount_);
    return;
  }
  if (rt.normal.z == 0) {
    adj("Wall", &wall_tex->specular_exponent_, &wall_tex->diffuse_ammount_);
    return;
  }
  adj("Ceiling", &ceiling_tex->specular_exponent_, &ceiling_tex->diffuse_ammount_);
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

void SoftwareRenderer::drawThread(int id) {
  float total = 0;
  vec3 xoffset = sight_x * fov;
  vec3 yoffset = sight_y * (fov * window_height_ / window_width_);
  vec3 dx = xoffset * (1.f/(window_width_ / 2));
  vec3 dy = yoffset * (1.f/(window_height_ / 2));

  Uint8* my_pixels = pixels_;
  BasePoint<double>* my_fppixels = fppixels_;
  int num_frames = frame_ - base_frame_;
  if ((num_frames & (num_frames - 1)) == 0 && id == 0 && num_frames > 16) {
    printf("Num frames: %d\n", num_frames);
    fflush(stdout);
  }
  double one_mul = 1. / num_frames / multiplier_;

  vec3 yray = sight - yoffset - xoffset;
  for (int y = 0; y < window_height_; y++) {
    if (y % numCPU_ == id) {
      vec3 ray = yray;
      for (int x = 0; x < window_width_; x++) {
        // no normalize here to preserve focal plane
        vec3 focused_ray = (ray + dx * antialiasing(gen) + dy * antialiasing(gen));
        vec3 focused_point = viewer + focused_ray * focused_distance;
        float r = sqrtf(lense_gen_r(gen)) * lense_blur;
        float a = lense_gen_a(gen);
        vec3 me = viewer + sight_x * (r * cos(a)) + sight_y * (r * sin(a));
        vec3 new_ray = normalize(focused_point - me);

        trace_values = x == 500 && y == 500;
        auto res = (max_depth == 0) ? trace_0(new_ray, me, 0) 
          : ((max_depth == 1) ? trace_1(new_ray, me, 0)
              : (max_depth == 2) ? trace_2(new_ray, me, 0)
              : trace_3(new_ray, me, 0));
        // accumulate
        *my_fppixels += BasePoint<double>::convert(res);
        res = BasePoint<float>::convert(*my_fppixels++ * one_mul);
        total += res.x + res.y + res.z;

//        vec3 saturated = saturateColor(res);
//        *my_pixels++ = colorToInt(saturated.z);
//        *my_pixels++ = colorToInt(saturated.y);
//        *my_pixels++ = colorToInt(saturated.x);
//        *my_pixels++ = 255;
        ray += dx;
      }
    } else {
      my_pixels += window_width_ * 4;
      my_fppixels += window_width_;
    }
    yray += dy;
  }
  screen_measure_[id] = total * multiplier_;
}

void SoftwareRenderer::postprocess() {
  int num_frames = frame_ - base_frame_;
  double one_mul = 1. / num_frames / multiplier_;

  for (int i = 0; i < window_width_ * window_height_; i++) {
    vec3 res = BasePoint<float>::convert(fppixels_[i] * one_mul);
    pp_fppixels_[i] = res;
  }
//  for (int y = 0; y < window_height_; y++) {
//    for (int x = 0; x < window_width_; x++) {
//      vec3 res = BasePoint<float>::convert(fppixels_[y * window_width_ + x] * one_mul);
//      float sum = res.x + res.y + res.z;
//      if (sum < 6) {
//        continue;
//      }
//      sum /= 3 * 200;
//      float fade = 0.90;
//      int n = (log(1./256/16) - log(sum)) / log(fade);
////      printf("%d %f\n", n, sum);
//      float mul = 0.005;
//      for (int i = x+1; i < std::min(window_width_, x+n); i++) {
//        pp_fppixels_[y * window_width_ + i] += res * mul;
//        mul *= fade;
//      }
//      mul = 0.005;
//      for (int i = x-1; i >= std::max(0, x-n); i--) {
//        pp_fppixels_[y * window_width_ + i] += res * mul;
//        mul *= fade;
//      }
//       mul = 0.005;
//      for (int i = y+1; i < std::min(window_height_, y+n); i++) {
//        pp_fppixels_[i * window_width_ + x] += res * mul;
//        mul *= fade;
//      }
//      mul = 0.005;
//      for (int i = y-1; i >= std::max(0, y-n); i--) {
//        pp_fppixels_[i * window_width_ + x] += res * mul;
//        mul *= fade;
//      }
//    }
//  }
  for (int i = 0; i < window_width_ * window_height_; i++) {
    vec3 res = pp_fppixels_[i];
    vec3 saturated = saturateColor(res);
    pixels_[i*4] = colorToInt(saturated.z);
    pixels_[i*4+1] = colorToInt(saturated.y);
    pixels_[i*4+2] = colorToInt(saturated.x);
    pixels_[i*4+3] = 255;
  }
}

void SoftwareRenderer::worker(int id) {
  int my_frame = 0;
  bool my_die;
  while (true) {
    my_frame++;
    {
      std::unique_lock<std::mutex> lk(m_);
      cv_.wait(lk, [&my_frame,this]{return my_frame == frame_;});
      my_die = die_;
    }
    if (!my_die) {
      drawThread(id);
    }
    {
      std::unique_lock<std::mutex> lk(m_);
      num_running_--;
      cv_.notify_all();
      if (my_die) return;
    }
  }
}

void SoftwareRenderer::draw() {
  auto inp = normalize(vec3(0,-1,-1));
  auto v = refract(1.9, vec3(0,0,1), inp);
  assert(v.x == 0);
  assert(v.size2() > 0.999 && v.size2() < 1.001);
  assert(std::abs(inp.y/v.y - 1.9) < 0.00001);


//  vec3 xmin = vec3(100, 100, 100), xmax = vec3(-100, -100, -100);
//  vec3 sz = vec3(ball_size, ball_size, ball_size);
//  for (int i = 0; i < LENGTH(balls); i++) {
//    const vec3& ball = balls[i].position_;
//    xmin = min(xmin, ball - sz);
//    xmax = max(xmax, ball + sz);
//  }
//  printf("Min: %f %f %f\n", xmin.x, xmin.y, xmin.z);
//  printf("Max: %f %f %f\n", xmax.x, xmax.y, xmax.z);

  if (threads_.size() == 0) {
    for (int i = 0; i < numCPU_; i++) {
      threads_.push_back(std::thread(&SoftwareRenderer::worker, this,i));
    }
    screen_measure_.resize(numCPU_);
  }

  {
    std::unique_lock<std::mutex> lk(m_);
    num_running_ = threads_.size();
    // Clear accumulated image
    if (frame_ == base_frame_) {
      memset(fppixels_, 0, window_width_ * window_height_ * sizeof(BasePoint<double>));
    }
    frame_++;
    cv_.notify_all();
  }
  {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [this]{return num_running_ == 0;});
  }
  postprocess();
  float total = 0;
  for (double m : screen_measure_) {
    total += m;
  }
  if (enable_exposure_compensation_) {
    multiplier_ = total / window_width_ / window_height_ / 3 * 10;
  }
  SDL_UpdateTexture(texture_, NULL, (void*)pixels_, window_width_ * sizeof(Uint32));
  SDL_RenderCopy(renderer_, texture_, NULL, NULL);
  SDL_RenderPresent(renderer_);
}

void SoftwareRenderer::reset_accumulate() {
  base_frame_ = frame_;
}

