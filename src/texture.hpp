#ifndef __TEXTURE_H__
#define __TEXTURE_H__

#include <memory>
#include <vector>
#include "vector.hpp"

class Texture {
  public:
    Texture() : width(-1), height(-1) {}
    static std::unique_ptr<Texture> Load(const std::string& path, float exp, float diff);
    std::tuple<vec3,vec3,float,float> Get(float x, float y, vec3 normal);
    float specular_exponent_;
    float diffuse_ammount_;
  private:
    int width;
    int height;
    // FIXME: memory leak
    std::vector<unsigned char> albedo;
    std::vector<unsigned char> normals;
    std::vector<unsigned char> roughness;
};

#endif  // __TEXTURE_H__
