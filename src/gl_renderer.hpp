#include <memory>
#define GL_GLEXT_PROTOTYPES 1
#include <GL/glcorearb.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

class OpenglRenderer {
  OpenglRenderer() = default;
  public:
  ~OpenglRenderer();

  static std::unique_ptr<OpenglRenderer> Create(int window_width, int window_height);
  bool setup();
  void draw();

  private:
  SDL_Window *window_ = nullptr;
  SDL_GLContext context_ = nullptr;
  int width_;
  int height_;

  GLuint ray_program;
  GLuint quad_vao;
  GLuint quad_program;
  GLuint tex_output;
};
