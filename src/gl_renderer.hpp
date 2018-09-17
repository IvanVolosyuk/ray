#include "vector.hpp"

#include <memory>
#include <vector>
#include <map>
#include <SDL2/SDL_opengl.h>
#include <GL/glew.h>


#include "renderer.hpp"
#include "texture.hpp"

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
  void bindTexture(int idx, Texture& tex);

  SDL_GLContext context_ = nullptr;
  int width_;
  int height_;
  std::map<int, std::vector<unsigned char>> textures;

  GLuint ray_program;
  GLuint quad_vao;
  GLuint quad_program;
  GLuint tex_output;

  std::vector<GLint> inputs_;
  GLint post_processor_mul_;
};
