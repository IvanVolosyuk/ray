#include <iostream>
#include <math.h>

#include "texture.hpp"
#include "png.hpp"

namespace {

void LoadImage(const std::string& path,
    unsigned char** bytes,
    int* width,
    int* height,
    bool expected_alpha) {
  int img_width;
  int img_height;
  bool img_alpha;
  bool res = loadPngImage(path.c_str(),
        &img_width, &img_height, &img_alpha, bytes);
  if (!res) {
    std::cerr << "Fail to load image: " << path << std::endl;
    std::exit(1);
  }
  if (*width == -1 && *height == -1)  {
    *width = img_width;
    *height = img_height;
  } else {
    if (img_width != *width || img_height != *height || expected_alpha != img_alpha) {
      std::cout << "Unexpected image: " << path << " "
        << img_width << "x" << img_height << " " << img_alpha;
      std::exit(1);
    }
  }
}

}  // namespace

std::unique_ptr<Texture> Texture::Load(const std::string& prefix, float exp) {
  auto t = std::make_unique<Texture>();
  t->width = t->height = -1;
  t->exp = exp;

  LoadImage(prefix + "_albedo.png", &t->albedo, &t->width, &t->height, false);
  LoadImage(prefix + "_normal.png", &t->normals, &t->width, &t->height, false);
  LoadImage(prefix + "_roughness.png", &t->roughness, &t->width, &t->height, false);
  return t;
}

std::tuple<vec3,vec3,float> Texture::Get(float x, float y, vec3 normal) {
    y = y - floor(y);
    if (isnan(y)) y = 0;
    x -= floor(x);
    if (isnan(x)) x = 0;
    int pos = ((int)(y * height) * width + (int)(x * width));
    vec3 color;
    color.x = albedo[pos*3 ] / 256.f;
    color.y = albedo[pos*3 + 1] / 256.f;
    color.z = albedo[pos*3 + 2] / 256.f;
    vec3 n;
    n.x = normals[pos*3] - 128;
    n.z = normals[pos*3+1] - 128;
    n.y = 128 - normals[pos*3+2];
    n = normalize(n);

    if (normal.z != 0) {
      std::swap(n.y, n.z);
      n.x *= 1;
      n.y *= 1;
      n.z *= -normal.z;
    } else if (normal.x != 0) {
      std::swap(n.x, n.y);
      n.x *= -normal.x;
    } else {
      n.y *= -normal.y;
    }
    float r = roughness[pos];
    float specular_exponent = 1 + exp * (256 - r);
    return std::make_tuple(color, n, specular_exponent);
}
