#include "vector.hpp"

#include <memory>
#include <vector>
#include <map>
#include <SDL2/SDL_opengl.h>
#include <GL/glew.h>
#include "imgui/imgui.h"

#include <optixu/optixpp_namespace.h>

#include "renderer.hpp"
#include "texture.hpp"

class OpenglRenderer : public Renderer {
  OpenglRenderer() = default;
  public:
  ~OpenglRenderer() override;

  static std::unique_ptr<Renderer> Create(
      int window_width, int window_height);
  bool setup();
  void initRenderer();
  void draw() override;
  virtual void reset_accumulate() override;
  bool WantCaptureMouse() override;
  bool WantCaptureKeyboard() override;
  void ProcessEvent(SDL_Event* event) override;

  private:
  void bindTexture(int idx, Texture& tex);

  SDL_GLContext context_ = nullptr;
  ImGuiIO* io_;
  int width_;
  int height_;
  std::map<int, std::vector<unsigned char>> textures;
  bool requireInit = true;

  GLuint quad_vao;
  GLuint quad_program;
  GLuint quad_pp_program;
  GLuint tex_output;

  GLint post_processor_mul_;
  optix::Context ctx_;
  optix::Program ray_prog_;
  optix::Program exc_prog_;
  optix::Buffer  bufferResult_;
  optix::Buffer  bufferAlbedo_;
  optix::Buffer  bufferNormal_;
  optix::Buffer  bufferTonemap_;
  optix::Buffer  bufferOutput_;
  optix::CommandList commandListTonemap_;
  optix::CommandList commandListDenoiser_;
  optix::Variable denoiseBlend_;
  optix::Variable exposure_;
  optix::Variable gamma_;

  optix::Buffer tree_buffer;
  optix::Buffer tri_buffer;
  optix::Buffer tri_lists_buffer;
};
