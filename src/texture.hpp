#ifndef __TEXTURE_H__
#define __TEXTURE_H__

#include <memory>
#include "vector.hpp"

class Texture {
  public:
    Texture() : width(-1), height(-1) {}
    static std::unique_ptr<Texture> Load(const std::string& path, float exp);
    std::tuple<vec3,vec3,float> Get(float x, float y, vec3 normal);
    float exp;
  private:
    int width;
    int height;
    unsigned char* albedo;
    unsigned char* normals;
    unsigned char* roughness;
};

#endif  // __TEXTURE_H__
