#include "vector.hpp"

#include <memory>
#include <vector>
#include <SDL2/SDL_opengl.h>
#include <GL/glew.h>


#include "renderer.hpp"

class OpenglRenderer : public Renderer {
  OpenglRenderer() = default;
  public:
  ~OpenglRenderer() override;

  static std::unique_ptr<Renderer> Create(
      int window_width, int window_height);
  bool setup();
  void draw() override;
  virtual void reset_accumulate() override;

  private:
  SDL_GLContext context_ = nullptr;
  int width_;
  int height_;

  GLuint ray_program;
  GLuint quad_vao;
  GLuint quad_program;
  GLuint tex_output;

  std::vector<GLint> inputs_;
  GLint post_processor_mul_;
};
