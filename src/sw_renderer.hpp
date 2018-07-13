#include <memory>
#include <SDL2/SDL.h>
#include "vector.hpp"
#include "renderer.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>

class SoftwareRenderer : public Renderer {
  SoftwareRenderer() = default;
  public:
  ~SoftwareRenderer() override;

  static std::unique_ptr<Renderer> Create(
      int window_width, int window_height);

  void draw() override;
  void reset_accumulate() override;

  static float distance(const vec3 norm_ray, const vec3 origin);

  private:
  void drawThread(int id);
  void worker(int id);
  void update_viewpoint();

  SDL_Renderer *renderer_ = nullptr;
  SDL_Texture * texture_ = nullptr;

  std::vector<std::thread> threads_;
  std::mutex m_;
  std::condition_variable cv_;
  int frame_ = 0;
  int base_frame_ = 0;
  bool die_ = false;
  int num_running_ = 0;
  BasePoint<double>* fppixels_;
  Uint8* pixels_;
  int numCPU_ = 0;
  int window_width_;
  int window_height_;
};
