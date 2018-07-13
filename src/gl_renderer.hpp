#include "vector.hpp"

#include <memory>
#define GL_GLEXT_PROTOTYPES 1
#include <GL/glcorearb.h>

#include <SDL2/SDL_opengl.h>

#include "renderer.hpp"

class OpenglRenderer : public Renderer {
  OpenglRenderer() = default;
  public:
  ~OpenglRenderer() override;

  static std::unique_ptr<Renderer> Create(
      int window_width, int window_height);
  bool setup();
  void draw() override;

  private:
  SDL_GLContext context_ = nullptr;
  int width_;
  int height_;

  GLuint ray_program;
  GLuint quad_vao;
  GLuint quad_program;
  GLuint tex_output;

  GLint viewer_location;
  GLint sight_location;
  GLint focused_distance_location;
};
