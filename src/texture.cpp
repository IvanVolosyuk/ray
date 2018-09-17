#include <iostream>
#include <math.h>

#include "texture.hpp"
#include "png.hpp"

namespace {

void LoadImage(const std::string& path,
    std::vector<unsigned char>* bytes,
    int* width,
    int* height,
    bool expected_alpha) {
  int img_width;
  int img_height;
  bool res = loadPngImage(path.c_str(),
        &img_width, &img_height, bytes);
  if (!res) {
    std::cerr << "Fail to load image: " << path << std::endl;
    std::exit(1);
  }
  if (*width == -1 && *height == -1)  {
    *width = img_width;
    *height = img_height;
  } else {
    if (img_width != *width || img_height != *height) {
      std::cout << "Unexpected image: " << path << " "
        << img_width << "x" << img_height << std::endl;
      std::exit(1);
    }
  }
}

}  // namespace

std::unique_ptr<Texture> Texture::Load(const std::string& prefix, float exp, float diff) {
  auto t = std::make_unique<Texture>();
  t->width = t->height = -1;
  t->specular_exponent_ = exp;
  t->diffuse_ammount_ = diff;

  LoadImage(prefix + "_albedo.png", &t->albedo, &t->width, &t->height, false);
  LoadImage(prefix + "_normal.png", &t->normals, &t->width, &t->height, false);
  LoadImage(prefix + "_roughness.png", &t->roughness, &t->width, &t->height, false);
  return t;
}

std::tuple<vec3,vec3,float,float> Texture::Get(float x, float y, vec3 normal) {
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
    return std::make_tuple(color, n, 1 + specular_exponent_ * (256 - r),
        diffuse_ammount_);
}

//struct Texture {
//  int width;
//  int height;
//  float specular_exponent;
//  float diffuse_ammount;
//  uint data[];
//};

const std::vector<unsigned char>& Texture::Export() {
  std::vector<unsigned char>& output = gl_export;
  output.resize(width * height * 8 * sizeof(int) + sizeof(float)*2 + sizeof(int) * 2);
  *(int*)&output[0] = width;
  *(int*)&output[4] = height;
  *(float*)&output[8] = specular_exponent_;
  *(float*)&output[12] = diffuse_ammount_;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      unsigned int r = albedo[(x + y * width) * 3 + 0];
      unsigned int g = albedo[(x + y * width) * 3 + 1];
      unsigned int b = albedo[(x + y * width) * 3 + 2];
      unsigned int nx = normals[(x + y * width) * 3 + 0];
      unsigned int ny = normals[(x + y * width) * 3 + 1];
      unsigned int nz = normals[(x + y * width) * 3 + 2];
      unsigned int rough = roughness[x + y * width];
      assert(r >= 0 && r < 256);
      assert(g >= 0 && g < 256);
      assert(b >= 0 && b < 256);
      assert(nx >= 0 && nx < 256);
      assert(ny >= 0 && ny < 256);
      assert(nz >= 0 && nz < 256);
      int res1 = (rough << 24) | (b << 16) | (g << 8) | r;
      int res2 = (nz << 16) | (ny << 8) | nx;
      *(unsigned int*)&output[16 + sizeof(int) * 2 * (y * width + x)] = res1;
      *(unsigned int*)&output[16 + sizeof(int) * 2 * (y * width + x) + 4] = res2;

//      *(float*)&output[16 + sizeof(int) * 8 * (y * width + a) + 4*0] = r / 256.;
//      *(float*)&output[16 + sizeof(int) * 2 * (y * width + x) + 4*1] = g / 256.;
//      *(float*)&output[16 + sizeof(int) * 2 * (y * width + x) + 4*2] = b / 256.;
//      *(float*)&output[16 + sizeof(int) * 2 * (y * width + x) + 4*3] = rough / 256.;
//      *(float*)&output[16 + sizeof(int) * 2 * (y * width + x) + 4*4] = (nx-128) / 256.;
//      *(float*)&output[16 + sizeof(int) * 2 * (y * width + x) + 4*5] = (ny-128) / 256.;
//      *(float*)&output[16 + sizeof(int) * 2 * (y * width + x) + 4*6] = (nz-128) / 256.;
//      *(float*)&output[16 + sizeof(int) * 2 * (y * width + x) + 4*7] = 0;
    }
  }
  return gl_export;
}
