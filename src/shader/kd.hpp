//#include <stdio.h>
//#include <vector>
//#include <cstdlib>
//#include <memory>
//#include <algorithm>
//#include <assert.h>
//#include <math.h>
#include "common.hpp"
#include <map>

#define AXE_X 0
#define AXE_Y 1
#define AXE_Z 2

//class vec3 {
//  public:
//    vec3() { v[0] = v[1] = v[2] = 0; }
//    vec3(const float v0[3]) { v[0] = v0[0]; v[1] = v0[1]; v[2] = v0[2]; }
//    vec3(float x, float y, float z) { v[0] = x; v[1] = y; v[2] = z; }
//    inline float operator[](int i) const { return v[i]; }
//    inline float& operator[](int i) { return v[i]; }
//  private:
//    float v[3];
//};

struct AABB {
  vec3 min;
  vec3 max;
  AABB combine(const AABB& x) {
    auto X = [&](int i) { return std::min(min[i], x.min[i]); };
    auto Y = [&](int i) { return std::max(max[i], x.max[i]); };
    return AABB{{X(0), X(1), X(2)}, {Y(0), Y(1), Y(2)}};
  }
};

std::vector<AABB> boxes;

struct tri_stl {
  vec3 normal;
  vec3 vertex[3];
};

struct tri {
  vec3 vertex[3];
  vec3 normal;
  vec3 vertex_normal[3];
  float inv_denom;

  tri() {}
  tri(tri_stl inp) {
    normal = inp.normal;
    vertex[0] = inp.vertex[0];
    vertex[1] = inp.vertex[1];
    vertex[2] = inp.vertex[2];
  }
};

std::vector<tri> tris;

class kd {
 public:
  int split_axe;
  float split_line;
  int child[2];
  std::vector<int> boxes;
};

struct kdtree {
  AABB bbox;
  std::vector<kd> item;
} kdtree;

struct Hit2 {
  int id;
  float distance;
  vec3 color;
  vec3 normal;
};


struct Ray {
  vec3 origin;
//  float dir[3];  // direction
  vec3 idir; // inverted direction
  vec3 dir;  // direction
  Hit2 traverse_recursive(int idx, float rmin, float rmax, bool front) const;
  Hit2 traverse_nonrecursive(int idx, float rmin, float rmax, bool front) const;
  bool intersect(AABB box, float* tmin_out) const;
  std::pair<float,float> intersect(AABB box) const;
  bool triangle_intersect(const tri& tr, float* t, float* u, float* v, bool front) const;
  static Ray make(vec3 origin, vec3 dir) {
    vec3 idir { 1.f/dir[0], 1.f/dir[1], 1.f/dir[2]};
    if (!isfinite(idir[0]) || idir[0] > 1e10 || idir[0] < -1e10) idir[0] = 1e10;
    if (!isfinite(idir[1]) || idir[1] > 1e10 || idir[1] < -1e10) idir[1] = 1e10;
    if (!isfinite(idir[2]) || idir[2] > 1e10 || idir[2] < -1e10) idir[2] = 1e10;

    return Ray { origin, idir, dir };
  }
};

struct StackEntry {
  int idx;
  float dist;
};

#define MAX_STACK 100
int64_t nhits = 0;
int64_t ntraverses = 0;
int64_t nintersects = 0;

bool Ray::triangle_intersect( 
    const tri& tr, 
    float* t, float* u, float* v, bool front) const { 
//  printf("[0]{%f %f %f}, [1]{%f %f %f}, [2]{%f %f %f}\norigin {%f %f %f} dir {%f %f %f}\n",
//      tr.vertex[0].x, tr.vertex[0].y, tr.vertex[0].z,
//      tr.vertex[1].x, tr.vertex[1].y, tr.vertex[1].z,
//      tr.vertex[2].x, tr.vertex[2].y, tr.vertex[2].z,
//      origin.x, origin.y, origin.z,
//      dir.x, dir.y, dir.z);

#define CULLING
//#define MOLLER_TRUMBORE
#ifdef MOLLER_TRUMBORE 
    vec3 v0v1 = tr.vertex[1] - tr.vertex[0]; 
    vec3 v0v2 = tr.vertex[2] - tr.vertex[0]; 
    vec3 pvec = cross(dir, v0v2); 
    float det = dot(v0v1, pvec); 
#ifdef CULLING 
    // if the determinant is negative the triangle is backfacing
    // if the determinant is close to 0, the ray misses the triangle
    if (det < kEpsilon) return false; 
#else 
    // ray and triangle are parallel if det is close to 0
    if (fabs(det) < kEpsilon) return false; 
#endif 
    float invDet = 1 / det; 
 
    vec3 tvec = origin - tr.vertex[0]; 
    *u = dot(tvec, pvec) * invDet; 
    if (*u < 0 || *u > 1) return false; 
 
    vec3 qvec = cross(tvec, v0v1); 
    *v = dot(dir, qvec) * invDet; 
    if (*v < 0 || *u + *v > 1) return false; 
 
    *t = dot(v0v2, qvec) * invDet; 
 
    return true; 
#else 
    vec3 N = tr.normal;
 
    // Step 1: finding P
 
    // check if ray and plane are parallel ?
    float NdotRayDirection = dot(N, dir); 
#ifdef CULLING 
    if (front ? NdotRayDirection > kEpsilon : NdotRayDirection < -kEpsilon) {
      P(NdotRayDirection);
      return false;
    }
#else
    if (fabs(NdotRayDirection) < kEpsilon) { // almost 0 
      P(NdotRayDirection);
//      printf("%f\n", NdotRayDirection);
        return false; // they are parallel so they don't intersect ! 
    }
#endif
 
    // compute t (equation 3)
    *t = (dot(N, tr.vertex[0] - origin)) / NdotRayDirection; 
    // check if the triangle is in behind the ray
    if (*t < 0) {
      P(*t);
      return false; // the triangle is behind 
    }
 
    // compute the intersection point using equation 1
    vec3 P = origin + dir * *t; 
 
    // Step 2: inside-outside test
    vec3 C; // vector perpendicular to triangle's plane 
 
    // edge 0
    vec3 edge0 = tr.vertex[1] - tr.vertex[0]; 
    vec3 vp0 = P - tr.vertex[0]; 
    C = cross(edge0, vp0); 
    if (dot(N, C) < 0) {
      P(vp0);
      P(edge0);
      P(N);
      return false; // P is on the right side 
    }
 
    // edge 1
    vec3 edge1 = tr.vertex[2] - tr.vertex[1]; 
    vec3 vp1 = P - tr.vertex[1]; 
    C = cross(edge1, vp1); 
    if ((*u = dot(N, C)) < 0) {
      P(vp1);
      P(edge1);
      P(N);
      return false; // P is on the right side 
    }
 
    // edge 2
    vec3 edge2 = tr.vertex[0] - tr.vertex[2]; 
    vec3 vp2 = P - tr.vertex[2]; 
    C = cross(edge2, vp2); 
    if ((*v = dot(N, C)) < 0) {
      P(vp2);
      P(edge2);
      P(N);
      return false; // P is on the right side; 
    }

    *u *= tr.inv_denom;
    *v *= tr.inv_denom;
 
    P(true);
    return true; // this ray hits the triangle 
#endif 
}

#define likely(x) __builtin_expect((x),1)
#define unlikely(x) __builtin_expect((x),0)

Hit2 Ray::traverse_recursive(int idx, float rmin, float rmax, bool front) const {
  ntraverses++;
  //printf("%d min %f max %f\n", idx, rmin, rmax);
  int axe = kdtree.item[idx].split_axe;
  if (likely(axe == -1)) {
    float dist = rmax;
    vec3 color;
    int hit = -1;
    //printf("node boxes: %d %ld\n", kdtree.item[idx].child[0], kdtree.item[idx].boxes.size());
    for (auto& box_id : kdtree.item[idx].boxes) {
      nintersects++;
//      printf("** %d", box_id);
      float new_dist, u, v;
      if (triangle_intersect(tris[box_id], &new_dist, &u, &v, front)) {
        
//        printf("* %d * %f vs %f rmax %f\n", box_id, new_dist, dist, rmax);
        if (new_dist < dist) {
          dist = new_dist;
          hit = box_id;
          color = vec3(1, u, v);
        }
      }
    }
    if (unlikely(hit != -1)) {
      nhits++;
//      printf("Found intersection with rmin=%f rmax=%f idx %d dist %f\n", rmin, rmax, idx, dist);
    } else {
//      printf("No intersection at rmin=%f rmax=%f idx %d\n", rmin, rmax, idx);
    }
    return {hit, dist, color};
  }
  float line = kdtree.item[idx].split_line;

  // line = origin + dir * dist
  // dist = (line - origin) * idir
  float dist = (line - origin[axe]) * idir[axe];
  int child_idx = idir[axe] < 0 ? 1 : 0;
//  printf("dist %f rmin %f rmax %f\n", dist, rmin, rmax);
  Hit2 hit;
  if (dist < rmin) {
    if ((hit = traverse_recursive(kdtree.item[idx].child[child_idx^1], rmin, rmax, front)).id != -1) return hit;
  } else if (dist > rmax) {
    if ((hit = traverse_recursive(kdtree.item[idx].child[child_idx], rmin, rmax, front)).id != -1) return hit;
  } else {
    if ((hit = traverse_recursive(kdtree.item[idx].child[child_idx], rmin, dist, front)).id != -1) return hit;
    if ((hit = traverse_recursive(kdtree.item[idx].child[child_idx^1], dist, rmax, front)).id != -1) return hit;
  }
  return {-1, rmax};
};

const float ray_epsilon = 1e-6;
const float construction_epsilon = 0;//1e-5f;

inline Hit2 Ray::traverse_nonrecursive(int idx, float rmin, float rmax, bool front) const {
  StackEntry stack[MAX_STACK];
//  int max_depth = reflect_gen(SW(gen)) * 10;
//  float orig_rmin = rmin;
//  int depth = 0;
  // TODO: Clamp rmin, rmax
  int stack_pos = 0;
  stack[stack_pos].idx = 0; // root
  stack[stack_pos].dist = rmax;
  stack_pos++;
  P("Trace nonrecursive");

  // needs rmin, takes rmax from stack
  while (likely(stack_pos > 0)) {
//    assert(stack_pos < MAX_STACK);
    stack_pos--;
    rmax = stack[stack_pos].dist;
    idx = stack[stack_pos].idx;

    while (true) {
      auto& item = kdtree.item[idx];
      int axe = item.split_axe;
      P(idx);
      P(rmin);
      P(rmax);
      P(axe);
//      depth++;
//      printf("Look into %d axe %d line %f rmin %f rmax %f\n",
//          idx, axe, item.split_line, rmin, rmax);
      if (likely(axe == -1)) {
        float dist = rmax + ray_epsilon;
        vec3 color(1);
        vec3 normal;
        int hit = -1;
//        printf("List of candidates: %ld\n", item.boxes.size());
        for (auto& box_id : item.boxes) {
//          printf("Candidate: %d\n", box_id);
          nintersects++;
          float new_dist, u, v;
          const auto& t = tris[box_id];
          P(box_id);
          if (likely(triangle_intersect(t, &new_dist, &u, &v, front))) {
            P(new_dist);
            P(dist);
            P(rmin);
            if (new_dist < rmin - ray_epsilon) {
              continue;
            }
            // if (new_dist > rmax) {
            //   if (!front) printf("Skip too far %f %f \n", new_dist, rmax);
            // }
            if (new_dist <= dist) {
              dist = new_dist;
              hit = box_id;
              //              color = vec3(u, v, 1-u-v);
              // FIXME: compute only for result
              normal = normalize(t.vertex_normal[0] * u + t.vertex_normal[1] * v + t.vertex_normal[2] * (1-u-v));
            }
          }
        }
        if (unlikely(hit != -1)) {
          nhits++;
          // FIXME
          P(hit);
          return {hit, dist, color, normal};
        }
        rmin = rmax - 2 * ray_epsilon;
        break;
      }
      float line = item.split_line;

      // line = origin + dir * dist
      // dist = (line - origin) * idir
      float dist = (line - origin[axe]) * idir[axe];
      int child_idx = idir[axe] < 0 ? 1 : 0;
      P(idir[axe]);
      P(child_idx);

      if (unlikely(dist < rmin - ray_epsilon)) {
        // push 1 rmin rmax
//        stack[stack_pos].dist = rmax;
//        stack[stack_pos++].idx = kdtree.item[idx].child[child_idx^1];
        idx = item.child[child_idx^1];
        P("before");
//        printf("dist %f < rmin %f : [%d] = %d (%f:%f)\n", dist, rmin, child_idx^1, idx, rmin, rmax);
      } else if (unlikely(dist > rmax + ray_epsilon)) {
        // push 0 rmin rmax
//        stack[stack_pos].dist = rmax;
//        stack[stack_pos++].idx = kdtree.item[idx].child[child_idx];
        idx = item.child[child_idx];
        P("after");
//        printf("dist %f > rmax %f : [%d] = %d (%f:%f)\n", dist, rmax, child_idx, idx, rmin, rmax);
      } else {
        P("both");
//      if (depth > max_depth & rmin > orig_rmin) {
//        vec3 normal;
//        normal[axe] = dir[axe] > 0 ? -1 : 1;
//        vec3 color;
//        color[axe] = 0.8;
//        return {1, rmin, normal, color};
//      }
        // push 1 dist rmax
//        printf("both : dist %f [%d] = %d (%f:%f), [%d] = %d (%f:%f)\n",
//            dist, child_idx, item.child[child_idx], rmin, dist,
//                  child_idx^1, item.child[child_idx^1], dist, rmax);
        stack[stack_pos].dist = rmax;
        stack[stack_pos++].idx = item.child[child_idx^1];
        // push 0 rmin dist
//        stack[stack_pos].dist = dist;
//        stack[stack_pos++].idx = kdtree.item[idx].child[child_idx];
        rmax = dist + ray_epsilon;
        idx = item.child[child_idx];
      }
    }
  }
  return {-1, max_distance};
};

//  float line;
//  int incr;
//  int box_idx;
//  int axe;
using Event = std::tuple<float, int, int, int>;

int other_axe[3][2] = {
  {1,2},
  {0,2},
  {1,2}
};

int build_tree(const AABB& bbox, const std::vector<Event>& events, int depth) {
  float epsilon = std::max(ray_epsilon, construction_epsilon);
  assert(depth < MAX_STACK - 2);
  float size[3];
  float cut_surface[3];
  int NL[3], NR[3], NP[3];
  for (int i = 0; i < 3; i++) {
    size[i] = bbox.max[i] - bbox.min[i];
    NL[i] = NP[i] = 0;
    NR[i] = events.size() / 6;
  }
  float max_surface = 0;
  for (int i = 0; i < 3; i++) {
    cut_surface[i] = size[other_axe[i][0]] * size[other_axe[i][1]];
    max_surface = std::max(max_surface, cut_surface[i]);
  }

  assert(max_surface > 0);
  for (int i = 0; i < 3; i++) {
    cut_surface[i] /= max_surface;
  }

  int best_axe = -1;
  // Cost:
  // 5.0: 14.40 FPS leafs: 847464
  // 2.0: 18.21 FPS leafs: 4933040
  // 1.0: 18.84 FPS leafs: 14184818
  // 0.5: 18.83 FPS leafs: 22386241
  //
  // Split penalty:
  // 2.0: 2x: 18.21 FPS leafs: 4933040 max: 105
  // 2.0: 3x: 17.38 FPS leafs: 1410756 max: 111
  // 1.0: 3x: 18.10 FPS leafs: 3143207
  // 0.5: 4x: 16.79 FPS leafs: 1511573 max: 184
  //
  // New scene, epsilons
  // 2.0, constr 0.00001, eps 0.00001  split 2: 9.51 FPS 4243642 leafs 105 max
  // 3.0, constr 0.00001, eps 0.00001  split 2: 9.30 FPS 2086528 leafs 105 max
  // 3.0, constr 0.00001, eps 0.       split 2: 9.55 FPS 2270260 leafs 105 max
  // 3.0, ray    0.000001, eps 0.      split 2: 9.92 FPS 1535519 leafs 102 max
  // 3.0, ray    0.00001, eps 0.       split 2: 9.92 FPS 1535519 leafs 102 max
  // 3.0, ray    0.00001, eps 0.00001  split 2: 9.83 FPS 1489430 leafs 103 max
  // 3.0, ray    1e-6,    eps 1e-6     split 2: 9.93 FPS 1517377 leafs 103 max
  // 2.0, ray    1e-6,    eps 1e-6     split 2: 10.29 FPS 2894953 leafs 103 max
  // 1.0, ray    1e-6,    eps 1e-6     split 2: 10.41 FPS 7016278 leafs 103 max
  //
  float split_cost = 2.0;  // versus triangle intersection
  float best_cost = 0.8 * events.size() / 6;
  float best_line = -1;
//  printf("Events: %ld\n", events.size());
//  printf("No split: {%f %f %f} {%f %f %f} cost: %f\n",
//      bbox.min[0], bbox.min[1], bbox.min[2],
//      bbox.max[0], bbox.max[1], bbox.max[2],
//      best_cost);

  // Find split point
  for (const Event& e : events) {
    float line = std::get<0>(e);
    int incr = std::get<1>(e);
    int axe = std::get<3>(e);
    if (incr == 1) {
      NL[axe]++;
      NP[axe]--;
    }
    if (line >= bbox.min[axe] && line <= bbox.max[axe]) {
      float left_area = (line - bbox.min[axe] + epsilon) / size[axe];
      float right_area = (bbox.max[axe] - line + epsilon) / size[axe];
      float min_area = 10 * epsilon / size[axe];
      float multiplier = NL[axe] == 0 || NR[axe] == 0 ? 0.8 : 1;
      float left_cost = split_cost + cut_surface[axe] * left_area * (NL[axe] + 2 * NP[axe]) * multiplier;
      float right_cost = split_cost + cut_surface[axe] * right_area * (NR[axe] + 2 * NP[axe]) * multiplier;
      float cost = std::max(left_cost, right_cost);
      //    printf("Area %f %f NL %d NR %d NP %d cost %f\n",
      //        left_area, right_area, NL[axe], NR[axe], NP[axe], cost);
      if (cost < best_cost && left_area > min_area && right_area > min_area) {
        //      printf("Posible split: axe %d cost %f line %f left_cost %f right_cost %f NL %d NR %d NP %d\n"
        //             "area %f %f\n\n",
        //          axe, cost, line, left_cost, right_cost, NL[axe], NR[axe], NP[axe], left_area, right_area);
        best_cost = cost;
        best_axe = axe;
        best_line = line;
      }
    }
    if (incr == -1) {
      NR[axe]--;
      NP[axe]++;
    }
  }

//  printf("Best line: %f axe: %d cost: %f\n", best_line, best_axe, best_cost);
  if (best_axe == -1) {
    // No split
    kd leaf {-1, 0, {(int)events.size(),-1}};
    for (const Event& e : events) {
      int axe = std::get<3>(e);
      int incr = std::get<1>(e);
      int box_idx = std::get<2>(e);
      if (axe == 0 && incr == -1) {
        leaf.boxes.push_back(box_idx);
      }
    }
//    printf("Leaf sz: %d %ld\n", leaf.child[0], leaf.boxes.size());
    kdtree.item.push_back(leaf);
    return kdtree.item.size() - 1;
  }

  std::vector<Event> left;
  std::vector<Event> right;
  // Split
  for (const Event& e : events) {
    int box_idx = std::get<2>(e);
    auto& b = boxes[box_idx];
    if (b.max[best_axe] <= best_line) {
      left.push_back(e);
    } else if (b.min[best_axe] >= best_line) {
      right.push_back(e);
    } else {
      left.push_back(e);
      right.push_back(e);
    }
  }
  assert(left.size() % 6 == 0);
  assert(right.size() % 6 == 0);
  AABB left_aabb = bbox, right_aabb = bbox;
  left_aabb.max[best_axe] = best_line;
  right_aabb.min[best_axe] = best_line;

  kdtree.item.push_back({best_axe, best_line, {-1, -1}});
  int res = kdtree.item.size() - 1;
//  printf("******* kd %d\n", res);
  int left_child = build_tree(left_aabb, left, depth+1);
  int right_child = build_tree(right_aabb, right, depth+1);
  kdtree.item[res].child[0] = left_child;
  kdtree.item[res].child[1] = right_child;
  return res;
}

void print_tree() {
  size_t max_tris = 0;
  size_t num_tri_refs = 0;
  size_t num_leafs = 0;
  std::vector<size_t> num_small_leafs = {0,0,0,0,0,0,0,0,0,0,0};
  printf("Tree: bbox: {%f %f %f} {%f %f %f}\n",
      kdtree.bbox.min.x,
      kdtree.bbox.min.y,
      kdtree.bbox.min.z,
      kdtree.bbox.max.x,
      kdtree.bbox.max.y,
      kdtree.bbox.max.z);
  for (size_t i = 0; i < kdtree.item.size(); i++) {
    const kd& k = kdtree.item[i];
    max_tris = std::max(max_tris, k.boxes.size());
    if (k.boxes.size() > 0) {
      num_leafs++;
      for (size_t j = 0; j < num_small_leafs.size(); j++) {
        if (k.boxes.size() == j) num_small_leafs[j]++;
      }
    }
    num_tri_refs += k.boxes.size();

//    printf("%ld: axe=%d, line=%f, left=%d right=%d [", i, k.split_axe, k.split_line,
//        k.child[0], k.child[1]);
//    for (size_t j = 0; j < k.boxes.size(); j++) {
//      printf("%d,", k.boxes[j]);
//    }
//    printf("]\n");
  }
//  for (size_t i = 0; i < boxes.size(); i++) {
//    const auto& b = boxes[i];
//    printf("box %ld: {%f %f %f} {%f %f %f}\n", i,
//        b.min.x, b.min.y, b.min.z,
//         b.max.x, b.max.y, b.max.z);
//  }
  printf("Max tris in a tree node: %ld avg %0.1f num leafs: %ld\n",
      max_tris, float(num_tri_refs) / num_leafs, num_leafs);
  for (size_t i = 0; i < num_small_leafs.size(); i++) {
    printf("Small leaf sz=%ld count %ld (%2.1f%%)\n",
        i, num_small_leafs[i], num_small_leafs[i] * 100.f / num_leafs);
  }
}

int build() {
  AABB bbox = boxes[0];
  for (size_t i = 1; i < boxes.size(); i++) {
    bbox = bbox.combine(boxes[i]);
  }
  kdtree.bbox = bbox;
  std::vector<Event> events;
  for (size_t i = 0; i < boxes.size(); i++) {
    const AABB& a = boxes[i];
    for (int x = 0; x < 3; x++) {
      events.push_back(Event{a.min[x], -1, i, x});
      events.push_back(Event{a.max[x], 1, i, x});
    }
  }
  std::sort(events.begin(), events.end());
  assert(build_tree(bbox, events, 0) == 0);
  print_tree();
  return 0;
}


std::pair<float,float> Ray::intersect(AABB box) const {
  // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
  // r.org is origin of ray
  float t1 = (box.min[0] - origin[0]) * idir[0];
  float t2 = (box.max[0] - origin[0]) * idir[0];
  float t3 = (box.min[1] - origin[1]) * idir[1];
  float t4 = (box.max[1] - origin[1]) * idir[1];
  float t5 = (box.min[2] - origin[2]) * idir[2];
  float t6 = (box.max[2] - origin[2]) * idir[2];
//  printf("{%f %f %f} {%f %f %f}\n", box.min[0], box.min[1], box.min[2], box.max[0], box.max[1], box.max[2]);
//  printf("%f %f %f %f %f %f\n", t1, t2, t3, t4, t5, t6);

  float tmin = std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
  float tmax = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));
  return std::make_pair(tmin, tmax);
}

bool Ray::intersect(AABB box, float* tmin) const {
  auto res = intersect(box);
//  printf("Interesect %f %f\n", res.first, res.second);
  if (res.first < 0 || res.first >= res.second) {
    return false;
  }
  *tmin = res.first;
  return true;
}

void gen2() {
  float epsilon = construction_epsilon;
  for (tri t : tris) {
    AABB bbox;
    bbox.min = t.vertex[0] - vec3(epsilon);
    bbox.max = t.vertex[0] + vec3(epsilon);
    for (int i = 1; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        bbox.min[j] = std::min(bbox.min[j], t.vertex[i][j] - epsilon);
        bbox.max[j] = std::max(bbox.max[j], t.vertex[i][j] + epsilon);
      }
    }
//    for (int i = 0; i < 3; i++) {
//      bbox.min[i] -= 0.0001;
//      bbox.max[i] += 0.0001;
//    }
    boxes.push_back(bbox);
  }

//  for (int i = 0; i < 1000; i++) {
//    float x = rand() % 1000;
//    float y = rand() % 1000;
//    float z = rand() % 1000;
//    float dx = rand() % 3 + 1;
//    float dy = rand() % 3 + 1;
//    float dz = rand() % 3 + 1;
//    boxes.push_back({{x,y,z}, {x+dx, y+dy, z+dz}});
////    printf("{%f %f %f} {%f %f %f}\n", x, y, z, x+dx, y+dy, z+dz);
//  }
}

void trace_parent(size_t kd, const tri& t) {
  if (kd == 0) {
    return;
  }
  for (size_t i = 0; i < kdtree.item.size(); i++) {
    auto& item = kdtree.item[i];

    int nleft = 0;
    int neq = 0;
    int nright = 0;
    for (int v = 0; v < 3; v++) {
      if (t.vertex[v][item.split_axe] < item.split_line) nleft++;
      if (t.vertex[v][item.split_axe] > item.split_line) nright++;
      if (t.vertex[v][item.split_axe] == item.split_line) neq++;
    }
    if (item.child[0] == (int)kd) {
      printf("Parent %ld axe %d line %f child[0] left %d eq %d right %d\n", i, item.split_axe, item.split_line,
          nleft, neq, nright);
      assert (nleft > 0);
      trace_parent(i, t);
      return;
    }
    if (item.child[1] == (int)kd) {
      printf("Parent %ld axe %d line %f child[1] left %d eq %d right %d\n", i, item.split_axe, item.split_line,
          nleft, neq, nright);
      assert (nright > 0);
      trace_parent(i, t);
      return;
    }
  }
  printf("Cannot find parent\n");
}

void find(int tri) {
  for (size_t i = 0; i < kdtree.item.size(); i++) {
    for (size_t t = 0; t < kdtree.item[i].boxes.size(); t++) {
      if (kdtree.item[i].boxes[t] == tri) {
        printf("Found tri %d at %ld box %ld of %ld\n", tri, i, t, kdtree.item[i].boxes.size());
        trace_parent(i, tris[tri]);
      }
    }
  }
}

void test_ray(const Ray& r) {
  int best = -1;
  float distance = 1e10;
  bool ambigious = false;
  for (size_t i = 0; i < tris.size(); i++) {
    float new_distance, u, v;
//    printf("%d: ", i);
    if (r.triangle_intersect(tris[i], &new_distance, &u, &v, true)) {
//      printf("Match: %f %f %f bbox {%f %f %f} {%f %f %f}\n", new_distance, u, v,
//          boxes[i].min[0], boxes[i].min[1], boxes[i].min[2],
//          boxes[i].max[0], boxes[i].max[1], boxes[i].max[2]);
      if (new_distance == distance) {
        ambigious = true;
      }
      if (new_distance >= 0 && new_distance < distance) {
        distance = new_distance;
        best = i;
      }
    }
  }
  int best_ray = -1;
  auto res = r.intersect(kdtree.bbox);
//  printf("Trace ray %f %f idir {%f %f %f}\n", res.first, res.second,
//      r.idir[0], r.idir[1], r.idir[2]);
  if (res.first <= res.second) {
    float tmin = std::max(0.f, res.first);
    float tmax = std::max(0.f, res.second);
    best_ray = r.traverse_nonrecursive(0, tmin, tmax, true).id;
  }
  if (ambigious) {
    assert(best_ray != -1);
    return;
  }
//  printf("%d vs %d \n", best, best_ray);
  if (best != best_ray && best_ray == -1) {
    find(best);
  }
  assert(best == best_ray);
}

vec3 normalize(vec3 v) {
  float length = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  return { v[0] / length, v[1] / length, v[2] / length };
}

void load_stl(const char* path) {
  FILE* f = fopen(path, "rb");
  if (f == nullptr) {
    std::string err = "Cannot open file" + std::string(path);
    perror(err.c_str());
    exit(1);
  }
  struct {
    char header[80];
    uint32_t size;
  } header;

  if (fread(&header, sizeof(header), 1, f) != 1) {
    printf("Wrong header: %s\n", path);
    exit(1);
  }
  if (header.size < 0 || header.size > 10000000) {
    printf("Wrong size: %d for %s\n", header.size, path);
    exit(1);
  }
  struct {
    tri_stl t;
    int16_t attributes;
  } record;
  static_assert(sizeof(tri_stl) == 4 * 3 * 4);
  int record_size = sizeof(tri_stl) + sizeof(int16_t);
  for (size_t i = 0; i < header.size; i++) {
    if (fread(&record, record_size, 1, f) != 1) {
      printf("Problem reading %ld triangle from %s\n", i, path);
      exit(1);
    }
    auto validate = [](float v) { assert(isfinite(v)); };
    assert(record.attributes == 0);
    for (int i = 0; i < 3; i++) {
      validate(record.t.normal[i]);
      for (int j = 0; j < 3; j++) {
        validate(record.t.vertex[j][i]);
//        record.t.vertex[j][i] /= 48;
      }
    }
    for (int i = 0; i < 3; i++) {
//      float z = record.t.vertex[i].z;
//      record.t.vertex[i].z = record.t.vertex[i].y;
//      record.t.vertex[i].y = -z;
//      record.t.vertex[i].z += 1.08;
    }

//      assert(record.t.normal.size2() > 0.99 && record.t.normal.size2() < 1.01);
      vec3 computed_normal = normalize(cross(record.t.vertex[1] - record.t.vertex[0],
            record.t.vertex[2] - record.t.vertex[0]));
//      assert(dot(record.t.normal, computed_normal) > 0);

      record.t.normal = computed_normal;
//      float sz = (computed_normal + normalize(record.t.normal)).size();
//      printf("%i: %f\n", i, sz);
//      assert((computed_normal + record.t.normal).size2() < 0.01f);
//      assert((record.t.normal + cross(record.t.vertex[2] - record.t.vertex[1],
//              record.t.vertex[1] -record.t.vertex[0])).size2() < 0.01f);
    tris.push_back(record.t);
  }
  assert(fread(&record, 1, 1, f) == 0);
  assert(feof(f));
  printf("Loaded %ld triangles\n", tris.size());

  std::map<std::tuple<float, float, float>, std::vector<vec3>> refs;
  for (size_t i = 0; i < tris.size(); i++) {
    const auto& t = tris[i];
    refs[std::make_tuple(t.vertex[0].x, t.vertex[0].y, t.vertex[0].z)].push_back(t.normal * cross(t.vertex[2] - t.vertex[0], t.vertex[1] - t.vertex[0]).size());
    refs[std::make_tuple(t.vertex[1].x, t.vertex[1].y, t.vertex[1].z)].push_back(t.normal * cross(t.vertex[0] - t.vertex[1], t.vertex[2] - t.vertex[1]).size());
    refs[std::make_tuple(t.vertex[2].x, t.vertex[2].y, t.vertex[2].z)].push_back(t.normal * cross(t.vertex[0] - t.vertex[2], t.vertex[1] - t.vertex[2]).size());
  }
  printf("Vertexes %ld unique %ld\n", tris.size() * 3, refs.size());
  for (size_t i = 0; i < tris.size(); i++) {
    auto& t = tris[i];
    for (int v = 0; v < 3; v++) {
      std::vector<vec3>& normals = refs[std::make_tuple(t.vertex[v].x, t.vertex[v].y, t.vertex[v].z)];
      vec3 res;
      for (const vec3& n : normals) {
        if (dot(t.normal, normalize(n)) > 0.50) {
          res += n;
        }
      }
      if (fabs(t.normal.x) == 1 || fabs(t.normal.y) == 1 || fabs(t.normal.z) == 1) {
        res = t.normal;
      }
      res = normalize(res);
      t.vertex_normal[v] = res;
    }
    vec3 c = cross(t.vertex[1] - t.vertex[0], t.vertex[2] - t.vertex[0]);
    t.inv_denom = 1./c.size();
  }
}

int64_t ntests = 300000;

void test_tri() {
//  printf("\n**********\n");
  tri t;
  t.normal = vec3(0, 0, 1);
  t.vertex[0] = vec3(1, 1, 5);
  t.vertex[1] = vec3(3, 1, 5);
  t.vertex[2] = vec3(3, 3, 5);
  t.vertex_normal[0] = normalize(t.vertex[0]);
  t.vertex_normal[1] = normalize(t.vertex[1]);
  t.vertex_normal[2] = normalize(t.vertex[2]);
  vec3 c = cross(t.vertex[1] - t.vertex[0], t.vertex[2] - t.vertex[0]);
  t.inv_denom = 1/c.size();
  printf("c %f inv_denom %f\n", c.size(), t.inv_denom);

  for (int i = 0; i < 100; i++) {
    float u0 = drand48();
    float v0 = drand48();
    if (u0 + v0 > 1) {
      printf(".");
      fflush(stdout);
      continue;
    }

    assert((t.vertex[0] * u0 + t.vertex[1] * v0 + t.vertex[2] * (1-u0-v0)).y
        == (t.vertex[0].y * u0 + t.vertex[1].y * v0 + t.vertex[2].y * (1-u0-v0)));

    Ray ray = Ray::make(vec3(0, 0, -5) + (t.vertex[0] * u0 + t.vertex[1] * v0 + t.vertex[2] * (1-u0-v0)), vec3(0, 0, 1));
    float dist, u, v;
    printf("u0 %f v0 %f origin %f %f %f idir %f %f %f\n", u0, v0, ray.origin.x, ray.origin.y, ray.origin.z, ray.idir.x, ray.idir.y, ray.idir.z); 
    assert(ray.triangle_intersect(t, &dist, &u, &v, false) == true);
    printf("u0 %f v0 %f dist: %f u: %f v: %f\n", u0, v0, dist, u, v); 
    assert(fabs(u - u0) < 0.001);
    assert(fabs(v - v0) < 0.001);
    assert(fabs(dist - 5) < 0.01);
//    vec3 n = normalize(t.vertex_normal[0] * u + t.vertex_normal[1] * v + t.vertex_normal[2] * (1-u-v));
  }
  printf("Done\n");
}
void test_rays() {
  test_tri();

  vec3 origin {200, 886.5, 778};
  vec3 idir { 1, 1e10f, 1e10f};
  Ray r { {origin[0], origin[1], origin[2]}, {idir[0], idir[1], idir[2]} };
  test_ray(r);
  vec3 center = (kdtree.bbox.max - kdtree.bbox.min) * 0.5f;
  for (int i = 0; i < ntests; i++) {

    vec3 origin {
      (float)drand48() * 4.f - 2.f,
      (float)drand48() * 4.f - 2.f,
      (float)drand48() * 4.f - 2.f
    };
    int axe = rand() % 3;
    float side = (rand() % 2) * 4 - 2.f + drand48() * 0.5 - 0.25;
    origin[axe] = side;
    vec3 dir = normalize(center - origin) + vec3 {
      (float)drand48() * 0.01f - 0.005f,
      (float)drand48() * 0.01f - 0.005f,
      (float)drand48() * 0.01f - 0.005f
    };
    dir = normalize(dir);
    Ray r = Ray::make(origin, dir);
//    printf("Ray: {%f %f %f} dir {%f %f %f}\n", origin[0], origin[1], origin[2], dir[0], dir[1], dir[2]);
    test_ray(r);
  }

}

