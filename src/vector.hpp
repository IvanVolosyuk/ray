#ifndef __VECTOR_H__
#define __VECTOR_H__

#include <assert.h>
#include <math.h>
#include <algorithm>

#ifndef M_PI
# define M_PI 3.14159265358979323846  /* pi */
#endif

#include <GL/glew.h>
#include <SDL2/SDL_opengl.h>

template<class T>
class BasePoint {
  public:
    using Value = T;
    T x, y, z;
    static const constexpr T w = 1;
    inline BasePoint() { x = y = z = 0; }
    inline explicit BasePoint(T v) { x = y = z = v; }

    inline BasePoint(const BasePoint<T>& v);

    inline BasePoint(T x, T y, T z);
    inline ~BasePoint() = default;
    void print() const;

    inline BasePoint& operator =(const BasePoint &v);
    inline BasePoint& operator +=(const BasePoint& v);
    inline BasePoint& operator -=(const BasePoint& v);
    inline BasePoint& operator *=(const T f);
    inline BasePoint operator -() const;
    inline bool operator == (const BasePoint& v) const;
    inline bool operator != (const BasePoint& v) const;
    inline T size() const;
    inline T size2() const;
    inline constexpr T operator[] (int idx) const {
      return idx == 0 ? x : ((idx == 1) ? y : z);
    }
    inline constexpr T& operator[] (int idx) {
      return idx == 0 ? x : ((idx == 1) ? y : z);
    }
    inline float sum() const;

    template<class Y>
    static inline BasePoint<T> convert(const BasePoint<Y>& from);
    operator const float*() const { return &x; }
};

using Point = BasePoint<float>;
using vec3 = Point;


class Matrix;

class Matrix2 {
    GLfloat data[16];
    public:

    Matrix2();
};


// opengl semantic matrix
// a[1] a[5] a[9]  a[13]
// a[2] a[6] a[10] a[14]
// a[3] a[7] a[11] a[15]
// a[4] a[8] a[12] a[16]

class Matrix {
    GLfloat data[16];
    public:
    GLfloat& operator() (int i, int j) { return data[(i - 1) + (j - 1) * 4]; }
    GLfloat operator() (int i, int j) const { return data[(i - 1) + (j - 1) * 4]; }
//    GLfloat* operator [](int x) { assert(x >= 0 && x <= 4); return data + 4 * x; }
//    const GLfloat* operator [](int x) const  { assert(x >= 0 && x <= 4); return data + 4 * x; }
    Matrix(GLfloat*);
    static Matrix& zero();
    static Matrix& identity();
    Matrix (Matrix &src);

    Matrix &move(Point& v);
    Matrix &rotate(GLfloat grad, Point& v);
    Matrix &mul(Matrix& m); /* mul this matrix on m */
    Matrix &invert();

    void print() const;

    friend void glMultMatrix(const Matrix& m);
    friend void glLoadMatrix(const Matrix& m);
    friend Point operator *(const Matrix &m, const Point& v);
};

template<class T>
BasePoint<T> operator -(const BasePoint<T>& p1, const BasePoint<T>& p2);
template<class T>
BasePoint<T> operator +(const BasePoint<T>& p, const BasePoint<T>& v);
template<class T>
BasePoint<T> operator -(const BasePoint<T>& p, const BasePoint<T>& v);
template<class T>
BasePoint<T> operator +(const BasePoint<T>& p, const BasePoint<T>& v);
template<class T>
BasePoint<T> operator -(const BasePoint<T>& p, const BasePoint<T>& v);
/* vector multiply */
template<class T>
BasePoint<T> cross(const BasePoint<T>& v1, const BasePoint<T>& v2);

template<class T>
T dot(const BasePoint<T>& v1, const BasePoint<T>& v2);

template<class T>
BasePoint<T> operator *(const BasePoint<T>& v1, const BasePoint<T>& v2);
template<class T>
BasePoint<T> operator /(const BasePoint<T>& v1, const BasePoint<T>& v2);

template<class T>
BasePoint<T> operator *(const BasePoint<T>& v, T f);
template<class T>
BasePoint<T> operator *(const Matrix &m, const BasePoint<T>& v);
template<class T>
BasePoint<T> operator *(const Matrix &m, const Point& v);

/***** INLINE FUNCTIONS ********/

template<class T>
inline BasePoint<T>::BasePoint(const T nx, const T ny, const T nz) {
    x = nx;
    y = ny;
    z = nz;
}

template<class T>
inline BasePoint<T>::BasePoint(const BasePoint<T>& p) {
    x = p.x;
    y = p.y;
    z = p.z;
}

template<class T>
template<class Y>
inline BasePoint<T> BasePoint<T>::convert(const BasePoint<Y>& p) {
  return BasePoint<T>(p.x, p.y, p.z);
}

template<class T>
inline BasePoint<T> BasePoint<T>::operator -() const {
    return BasePoint<T>(-x, -y, -z);
}

template<class T>
inline BasePoint<T>& BasePoint<T>::operator +=(const BasePoint<T>& v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

template<class T>
inline BasePoint<T>& BasePoint<T>::operator -=(const BasePoint<T>& v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
}

template<class T>
inline BasePoint<T>& BasePoint<T>::operator =(const BasePoint<T>& p) {
    x = p.x;
    y = p.y;
    z = p.z;
    return *this;
}

template<class T>
inline BasePoint<T>& BasePoint<T>::operator *=(const T f) {
    x *= f;
    y *= f;
    z *= f;
    return *this;
}

template<class T>
inline T BasePoint<T>::size2() const {
    return x*x + y*y + z*z;
}

template<class T>
inline T BasePoint<T>::size() const {
    return sqrt(x*x + y*y + z*z);
}

template<class T>
inline BasePoint<T> normalize(BasePoint<T> v) {
    T sz = v.size();
    sz = BasePoint<T>::w/sz;
    BasePoint<T> res(v.x * sz, v.y * sz, v.z * sz);
    return res;
}

template<class T>
inline BasePoint<T> operator -(const BasePoint<T>& p1, const BasePoint<T>& p2) {
    BasePoint<T> res(p1.x - p2.x,
	             p1.y - p2.y,
	             p1.z - p2.z);
    return res;
}


template<class T>
inline BasePoint<T> operator +(const BasePoint<T>& p, const BasePoint<T>& v) {
    BasePoint<T> res(p);
    return res+=v;
}

template<class T>
inline BasePoint<T> operator *(const BasePoint<T>& v1, const BasePoint<T>& v2) {
    BasePoint<T> res = BasePoint<T>(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
    return res;
}
template<class T>
inline BasePoint<T> operator /(const BasePoint<T>& v1, const BasePoint<T>& v2) {
    BasePoint<T> res = BasePoint<T>(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
    return res;
}

template<class T>
inline T dot(const BasePoint<T>& v1, const BasePoint<T>& v2) {
    T res = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    return res;
}

template<class T>
inline BasePoint<T> cross(const BasePoint<T>& v1, const BasePoint<T>& v2) {
    BasePoint<T> res(v1.y * v2.z - v1.z * v2.y,
	     -(v1.x * v2.z - v1.z * v2.x),
               v1.x * v2.y - v1.y * v2.x);
    return res;
}


template<class T>
inline float BasePoint<T>::sum() const {
    return x + y + z;
}

template<class T>
inline BasePoint<T> operator *(const BasePoint<T>& v, T f) {
    BasePoint<T> res(v);
    return res*=f;
}

template<class T>
inline bool
BasePoint<T>::operator == (const BasePoint<T>& v) const
{
    return ((x > v.x * 0.99) || (x < v.x * 1.01))
        && ((y > v.y * 0.99) || (y < v.y * 1.01))
        && ((z > v.z * 0.99) || (z < v.z * 1.01));
}

template<class T>
inline bool
BasePoint<T>::operator != (const BasePoint<T>& v) const
{
    return (x != v.x) || (y != v.y) || (z != v.z);
}

template<class T>
BasePoint<T> min(const BasePoint<T>& v1, const BasePoint<T>& v2) {
  return BasePoint<T>(std::min(v1.x, v2.x), std::min(v1.y, v2.y), std::min(v1.z, v2.z));
}

template<class T>
BasePoint<T> max(const BasePoint<T>& v1, const BasePoint<T>& v2) {
  return BasePoint<T>(std::max(v1.x, v2.x), std::max(v1.y, v2.y), std::max(v1.z, v2.z));
}

#endif // __VECTOR_H__
