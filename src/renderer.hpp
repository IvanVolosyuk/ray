#ifndef __RENDERER_H__
#define __RENDERER_H__

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include "texture.hpp"

class Renderer {
  public:
  Renderer() = default;
  virtual ~Renderer() = default;
  
  SDL_Window* GetWindow() { return window_; }

  virtual void draw() {}
  virtual void reset_accumulate() {}
  virtual void set_exposure_compensation(bool e) {}
  virtual bool WantCaptureMouse() { return false; }
  virtual bool WantCaptureKeyboard() { return false; }
  virtual void ProcessEvent(SDL_Event* event) {};

  protected:
  SDL_Window *window_ = nullptr;
};

extern std::unique_ptr<Texture> wall_tex;
extern std::unique_ptr<Texture> ceiling_tex;
extern std::unique_ptr<Texture> floor_tex;

#endif  // __RENDERER_H__
