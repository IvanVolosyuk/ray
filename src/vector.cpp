#include <stdio.h>
#include <assert.h>

#include "vector.hpp"

template<>
void BasePoint<float>::print() const {
    printf("Point(%f, %f, %f)", x, y, z);
    fflush(stdout);
}

template<>
void BasePoint<double>::print() const {
    printf("Point(%f, %f, %f)", x, y, z);
    fflush(stdout);
}

GLfloat _zero_data[] = {
    0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 0.f
};

GLfloat _identity_data[] = {
    1.f, 0.f, 0.f, 0.f,
    0.f, 1.f, 0.f, 0.f,
    0.f, 0.f, 1.f, 0.f,
    0.f, 0.f, 0.f, 1.f
};

Matrix _zero(_zero_data);
Matrix _identity(_identity_data);

Matrix& Matrix::zero() {
    return _zero;
}

Matrix& Matrix::identity() {
    return _identity;
}

Matrix::Matrix(GLfloat *src) {
    for(int i = 0; i < 16; i++) data[i] = src[i];
}

Matrix::Matrix(Matrix &src) {
    for(int i = 0; i < 16; i++) data[i] = src.data[i];
}

void
Matrix::print() const {
    printf("Matrix(%9.2f, %9.2f, %9.2f %9.2f)\n", data[0], data[4], data[8],  data[12]);
    printf("      (%9.2f, %9.2f, %9.2f %9.2f)\n", data[1], data[5], data[9],  data[13]);
    printf("      (%9.2f, %9.2f, %9.2f %9.2f)\n", data[2], data[6], data[10], data[14]);
    printf("      (%9.2f, %9.2f, %9.2f %9.2f)\n", data[3], data[7], data[11], data[15]);
}

Point
operator *(const Matrix &m, const Point& v) {
    Point res;
    GLfloat dist;
    res.x = v.x * m(1,1) + v.y * m(1,2) + v.z * m(1,3) + v.w * m(1,4);
    res.y = v.x * m(2,1) + v.y * m(2,2) + v.z * m(2,3) + v.w * m(2,4);
    res.z = v.x * m(3,1) + v.y * m(3,2) + v.z * m(3,3) + v.w * m(3,4);
    dist =  v.x * m(4,1) + v.y * m(4,2) + v.z * m(4,3) + v.w * m(4,4);
    res *= 1./dist;
    return res;
}

Matrix&
Matrix::move(Point& v) {
    Matrix m(Matrix::identity());
    m(1,4) = v.x;
    m(2,4) = v.y;
    m(3,4) = v.z;
    mul(m);
//    printf("[J[H\n");
//    print();
    return *this;
}
Matrix&
Matrix::rotate(GLfloat grad, Point& v) {
    GLfloat rad = grad / 180. * M_PI;
    GLfloat COS = cos(rad);
    GLfloat SIN = sin(rad);

    Matrix res(Matrix::identity());
    Point u(v); u.normalize();

#define F(k,x,y) u.x*u.y + COS*(k-u.x*u.y)
    res(1,1) = F(1,x,x);
    res(2,1) = F(0,x,y) + SIN * u.z;
    res(3,1) = F(0,x,z) - SIN * u.y;

    res(1,2) = F(0,y,x) - SIN * u.z;
    res(2,2) = F(1,y,y);
    res(3,2) = F(0,y,z) + SIN * u.x;

    res(1,3) = F(0,z,x) + SIN * u.y;
    res(2,3) = F(0,z,y) - SIN * u.x;
    res(3,3) = F(1,z,z);
    mul(res);
    return *this;
}

Matrix&
Matrix::mul(Matrix& m2) {
    Matrix m1(*this);
    for(int x = 0; x < 4; x++)
        for(int y = 0; y < 4; y++) {
            GLfloat res = 0;
            for(int i = 0; i < 4; i++)
                res += m1(i+1,x+1) * m2(y+1,i+1);
            operator () (y+1,x+1) = res;
        }
    return *this;
}

/*static int find(int *arr, int x) {
    for(int i = 0; i < 4; i++)
        if (arr[i] == x) return i;
    assert(0);
    return 0;
}*/

GLfloat determinant(Matrix& m) {
    //printf("\n");
    //m.print();
    GLfloat sum = 0;
    int sign0 = 1, sign1, sign2;
    for(int i0 = 0; i0 < 4; i0++) {
        sign1 = sign0;
        for(int i1 = 0; i1 < 4; i1++) {
            if (i1 == i0) continue;
            sign2 = sign1;
            for(int i2 = 0; i2 < 4; i2++) {
                if (i2 == i0 || i2 == i1) continue;
                for(int i3 = 0; i3 < 4; i3++) {
                    if (i3 == i0 || i3 == i1 || i3 == i2) continue;
                    /* internal check, number of disorders into sign:
                    int x[4] = {i0, i1, i2, i3};
                    int sign = 1;

                    if (x[0] != 0) { int idx = find(x,0); int tmp = x[idx]; x[idx] = x[0]; x[0] = tmp; sign *= -1; }
                    if (x[1] != 1) { int idx = find(x,1); int tmp = x[idx]; x[idx] = x[1]; x[1] = tmp; sign *= -1; }
                    if (x[2] != 2) { int idx = find(x,2); int tmp = x[idx]; x[idx] = x[2]; x[2] = tmp; sign *= -1; }
                    if (x[3] != 3) { int idx = find(x,3); int tmp = x[idx]; x[idx] = x[3]; x[3] = tmp; sign *= -1; }
                    assert(sign2 == sign);*/

                    sum += sign2 * m(i0+1,1) * m(i1+1,2) * m(i2+1,3) * m(i3+1,4);
                    //printf("%i %i %i %i sign %i --\n", i0, i1, i2, i3, sign2);
                }
                sign2 *= -1;
            }
            sign1 *= -1;
        }
        sign0 *= -1;
    }
    return sum;
}

GLfloat Minor(Matrix& m, int x, int y) {
    int idx[3];
    int i, j;
    for(i = 0, j = 0; i < 4; i++) {
        if(i == x) continue;
        idx[j++] = i;
    }
    GLfloat sum = 0;
    int sign0 = 1, sign1;
    for(int i0 = 0; i0 < 4; i0++) {
        if (i0 == y) continue;
        sign1 = sign0;
        for(int i1 = 0; i1 < 4; i1++) {
            if (i1 == y || i1 == i0) continue;
            for(int i2 = 0; i2 < 4; i2++) {
                if (i2 == y || i2 == i0 || i2 == i1) continue;
                sum += sign1 * m(i0+1,idx[0]+1)
                             * m(i1+1,idx[1]+1)
                             * m(i2+1,idx[2]+1);

                //printf("%i %i %i sign %i\n", i0, i1, i2, sign1);
            }
            sign1 *= -1;
        }
        sign0 *= -1;
    }
    return sum;
}

Matrix&
Matrix::invert() {
    Matrix res = Matrix::zero();
    GLfloat determ = determinant(*this);
    assert(determ != 0);
    for(int x = 0; x < 4; x++)
    for(int y = 0; y < 4; y++)
        res(y+1,x+1) = ((x + y) & 1 ? -1 : 1) * Minor(*this, y, x) / determ;

    return *this = res;
}


