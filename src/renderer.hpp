#ifndef __RENDERER_H__
#define __RENDERER_H__

#include <GL/glew.h>
#include <SDL2/SDL.h>

class Renderer {
  public:
  Renderer() = default;
  virtual ~Renderer() = default;
  
  SDL_Window* GetWindow() { return window_; }

  virtual void draw() {}
  virtual void reset_accumulate() {}
  virtual void set_exposure_compensation(bool e) {}

  protected:
  SDL_Window *window_ = nullptr;
};

#endif  // __RENDERER_H__
