#ifndef __VECTOR_H__
#define __VECTOR_H__

#define GL_GLEXT_PROTOTYPES
#include <GL/glut.h>
#include <GL/glext.h>
#include <assert.h>
#include <math.h>

#ifndef M_PI
# define M_PI 3.14159265358979323846  /* pi */
#endif

class Point {
    public:
    GLfloat x, y, z;
    static const constexpr GLfloat w = 1;
    inline Point() { x = y = z = 0; }
    inline Point(const Point& v);
    inline Point(GLfloat x, GLfloat y, GLfloat z);
    void print() const;

    inline Point& operator =(const Point &v);
    inline Point& operator +=(const Point& v);
    inline Point& operator -=(const Point& v);
    inline Point& operator *=(const GLfloat f);
    inline Point operator -() const;
    inline bool operator == (const Point& v) const;
    inline bool operator != (const Point& v) const;
    inline GLfloat size() const;
    inline GLfloat size2() const;
    inline Point normalize() const;
    inline Point mul(const Point& v) const;
    inline float sum() const;

    friend inline Point operator ^(const Point& v1, const Point& v2);
    friend inline GLfloat operator *(const Point& v1, const Point& v2);
};

class Matrix;
static inline void glMultMatrix(const Matrix& m);
static inline void glLoadMatrix(const Matrix& m);

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

Point operator -(const Point& p1, const Point& p2);
Point  operator +(const Point& p, const Point& v);
Point  operator -(const Point& p, const Point& v);
Point operator +(const Point& p, const Point& v);
Point operator -(const Point& p, const Point& v);
/* vector multiply */
Point operator ^(const Point& v1, const Point& v2);
/* scalar multiply */
GLfloat operator *(const Point& v1, const Point& v2);
Point operator *(const Point& v, GLfloat f);
Point operator *(const Matrix &m, const Point& v);
Point operator *(const Matrix &m, const Point& v);

/***** INLINE FUNCTIONS ********/

/* GL specific */
static inline void
glNormal(Point n) {
    glNormal3f(n.x, n.y, n.z);
}

static inline void
glVertex(Point p) {
    glVertex3f(p.x, p.y, p.z);
}

static inline void
glColor(Point c) {
    glColor3f(c.x, c.y, c.z);
}

static inline void
glTranslate(Point t) {
    glTranslatef(t.x, t.y, t.z);
}
static inline void
glMultMatrix(const Matrix& m) {
    glMultMatrixf(m.data);
}

static inline void
glLoadMatrix(const Matrix& m) {
    glLoadMatrixf(m.data);
}

inline Point::Point(GLfloat nx, GLfloat ny, GLfloat nz) {
    x = nx;
    y = ny;
    z = nz;
}

inline Point::Point(const Point& p) {
    x = p.x;
    y = p.y;
    z = p.z;
}

inline Point
Point::operator -() const {
    return Point(-x, -y, -z);
}

inline Point&
Point::operator +=(const Point& v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

inline Point&
Point::operator -=(const Point& v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
}

inline Point&
Point::operator =(const Point& p) {
    x = p.x;
    y = p.y;
    z = p.z;
    return *this;
}

inline Point&
Point::operator *=(const GLfloat f) {
    x *= f;
    y *= f;
    z *= f;
    return *this;
}

inline GLfloat
Point::size2() const {
    return x*x + y*y + z*z;
}

inline GLfloat
Point::size() const {
    return sqrt(x*x + y*y + z*z);
}

inline Point
Point::normalize() const {
    GLfloat sz = size();
    sz = 1.f/sz;
    Point res(x * sz, y * sz, z * sz);
    return res;
}

inline Point
operator -(const Point& p1, const Point& p2) {
    Point res(p1.x - p2.x,
	       p1.y - p2.y,
	       p1.z - p2.z);
    return res;
}


inline Point
operator +(const Point& p, const Point& v) {
    Point res(p);
    return res+=v;
}

inline GLfloat
operator *(const Point& v1, const Point& v2) {
    GLfloat res = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    return res;
}
inline Point
operator ^(const Point& v1, const Point& v2) {
    Point res(v1.y * v2.z - v1.z * v2.y,
	     -(v1.x * v2.z - v1.z * v2.x),
               v1.x * v2.y - v1.y * v2.x);
    return res;
}

inline Point Point::mul(const Point& v) const {
    return Point(x * v.x, y * v.y, z * v.z);
}

inline float Point::sum() const {
    return x + y + z;
}

inline Point
operator *(const Point& v, GLfloat f) {
    Point res(v);
    return res*=f;
}

inline bool
Point::operator == (const Point& v) const
{
    return ((x > v.x * 0.99) || (x < v.x * 1.01))
        && ((y > v.y * 0.99) || (y < v.y * 1.01))
        && ((z > v.z * 0.99) || (z < v.z * 1.01));
}
inline bool
Point::operator != (const Point& v) const
{
    return (x != v.x) || (y != v.y) || (z != v.z);
}

#endif // __VECTOR_H__
