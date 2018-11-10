
struct AABB {
  vec3 min;
  vec3 max;
  AABB combine(const AABB& x) {
    auto X = [&](int i) { return std::min(min[i], x.min[i]); };
    auto Y = [&](int i) { return std::max(max[i], x.max[i]); };
    return AABB{{X(0), X(1), X(2)}, {Y(0), Y(1), Y(2)}};
  }
};

struct tri_stl {
  vec3 normal;
  vec3 vertex[3];
};

struct tri {
  vec3 vertex[3];
  vec3 normal;
  vec3 vertex_normal[3];

  tri() {}
  tri(tri_stl inp) {
    normal = inp.normal;
    vertex[0] = inp.vertex[0];
    vertex[1] = inp.vertex[1];
    vertex[2] = inp.vertex[2];
  }
};

const int MAX_EMBEDDED = 0;
class kd {
 public:
  int split_axe_and_idx;
  float split_line;
};

struct kdtree {
  AABB bbox;
  std::vector<kd> item;
};

extern struct kdtree kdtree;
