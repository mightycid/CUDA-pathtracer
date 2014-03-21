/**
 * Copyright (C) 2014 MightyCid
 *
 * This file is part of CUDA-pathtracer <github.com/mightycid/CUDA-pathtracer>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>. 
 */

#ifndef __VECMATH_H__
#define __VECMATH_H__

/**
 * Represents a 3D vector
 * 4th value for compatibility with matrices.
 * It isn't used in operations but it is accessible
 */
struct Vec {
	inline CUDA_HOST_DEVICE Vec() { x = y = z = w = 0.f; }
	inline CUDA_HOST_DEVICE Vec(float x_, float y_=0.f, float z_=0.f, float w_=0.f)
		: x(x_), y(y_), z(z_), w(w_) { }
	inline CUDA_HOST_DEVICE Vec(const Vec &v) : x(v.x), y(v.y), z(v.z), w(v.w) { }

	inline CUDA_HOST_DEVICE float operator[](int i) const {
		assert(i >= 0 && i <= 3); return (&x)[i];
	}
	inline CUDA_HOST_DEVICE float& operator[](int i) {
		assert(i >= 0 && i <= 3); return (&x)[i];
	}

	inline CUDA_HOST_DEVICE Vec operator-() const {
		return Vec(-x, -y, -z, -w);
	}

	inline CUDA_HOST_DEVICE Vec Cross(const Vec &v) const {
		return Vec(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
	}
	inline CUDA_HOST_DEVICE float Dot(const Vec &v) const {
		return x*v.x + y*v.y + z*v.z;
	}
	inline CUDA_HOST_DEVICE float Length() const {
		return sqrtf(x*x + y*y + z*z);
	}
	inline CUDA_HOST_DEVICE float LengthSquared() const {
		return x*x + y*y + z*z;
	}
	inline CUDA_HOST_DEVICE Vec& Normalize();
	inline CUDA_HOST_DEVICE Vec& Normalize(float *l);

	float x, y, z, w;
};

inline CUDA_HOST_DEVICE Vec& operator+=(Vec &v1, const Vec &v2) {
	v1.x += v2.x; v1.y += v2.y; v1.z += v2.z; return v1;
}
inline CUDA_HOST_DEVICE Vec& operator+=(Vec &v, float d) {
	v.x += d; v.y += d; v.z += d; return v;
}

inline CUDA_HOST_DEVICE Vec& operator-=(Vec &v1, const Vec &v2) {
	v1.x -= v2.x; v1.y -= v2.y; v1.z -= v2.z; return v1;
}
inline CUDA_HOST_DEVICE Vec& operator-=(Vec &v, float d) {
	v.x -= d; v.y -= d; v.z -= d; return v;
}

inline CUDA_HOST_DEVICE Vec& operator*=(Vec &v1, const Vec &v2) {
	v1.x *= v2.x; v1.y *= v2.y; v1.z *= v2.z; return v1;
}
inline CUDA_HOST_DEVICE Vec& operator*=(Vec &v, float d) {
	v.x *= d; v.y *= d; v.z *= d; return v;
}

inline CUDA_HOST_DEVICE Vec& operator/=(Vec &v1, const Vec &v2) {
	v1.x /= v2.x; v1.y /= v2.y; v1.z /= v2.z; return v1;
}
inline CUDA_HOST_DEVICE Vec& operator/=(Vec &v, float d) {
	assert(d != 0.f); const float inv = 1.f / d; return v *= inv;
}

inline CUDA_HOST_DEVICE Vec operator+(const Vec &v1, const Vec &v2) {
	return Vec(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}
inline CUDA_HOST_DEVICE Vec operator+(const Vec &v, float d) {
	return Vec(v.x + d, v.y + d, v.z + d);
}
inline CUDA_HOST_DEVICE Vec operator+(float d, const Vec &v) {
	return Vec(v.x + d, v.y + d, v.z + d);
}

inline CUDA_HOST_DEVICE Vec operator-(const Vec &v1, const Vec &v2) {
	return Vec(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}
inline CUDA_HOST_DEVICE Vec operator-(const Vec &v, float d) {
	return Vec(v.x - d, v.y - d, v.z - d);
}
inline CUDA_HOST_DEVICE Vec operator-(float d, const Vec &v) {
	return Vec(v.x - d, v.y - d, v.z - d);
}

inline CUDA_HOST_DEVICE Vec operator*(const Vec &v1, const Vec &v2) {
	return Vec(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z);
}
inline CUDA_HOST_DEVICE Vec operator*(const Vec &v, float d) {
	return Vec(v.x*d, v.y*d, v.z*d);
}
inline CUDA_HOST_DEVICE Vec operator*(float d, const Vec &v) {
	return Vec(v.x*d, v.y*d, v.z*d);
}

inline CUDA_HOST_DEVICE Vec operator/(const Vec &v1, const Vec &v2) {
	return Vec(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}
inline CUDA_HOST_DEVICE Vec operator/(const Vec &v, float d) {
	assert(d != 0.f); const float inv = 1.f / d; return v*inv;
}
inline CUDA_HOST_DEVICE Vec operator/(float d, const Vec &v) {
	assert(d != 0.f); const float inv = 1.f / d; return v*inv;
}

inline CUDA_HOST_DEVICE Vec& Vec::Normalize() {
	return *this = *this / Length();
}
inline CUDA_HOST_DEVICE Vec& Vec::Normalize(float *l) {
	*l = Length(); return *this = *this / *l;
}

/**
 * Represents a 3D-Point in space
 */
struct Point {
	inline CUDA_HOST_DEVICE Point() { x = y = z = 0.f; }
	inline CUDA_HOST_DEVICE Point(float x_, float y_=0.f, float z_=0.f)
		: x(x_), y(y_), z(z_) { }
	inline CUDA_HOST_DEVICE Point(const Point &p) : x(p.x), y(p.y), z(p.z) { }

	inline CUDA_HOST_DEVICE Point& operator=(const Point &p) {
		x = p.x; y = p.y; z = p.z; return *this;
	}
	inline CUDA_HOST_DEVICE float operator[](int i) const {
		assert(i >= 0 && i <= 3); return (&x)[i];
	}
	inline CUDA_HOST_DEVICE float& operator[](int i) {
		assert(i >= 0 && i <= 3); return (&x)[i];
	}

	inline CUDA_HOST_DEVICE bool operator==(const Point &p) const {
		return x == p.x && y == p.y && z == p.z;
	}
	inline CUDA_HOST_DEVICE bool operator!=(const Point &p) const {
		return x != p.x || y != p.y || z != p.z;
	}

	inline CUDA_HOST_DEVICE Point operator-() const {
		return Point(-x, -y, -z);
	}

	inline CUDA_HOST_DEVICE float Distance(const Point &p) const;
	inline CUDA_HOST_DEVICE float DistanceSquared(const Point &p) const;

	float x, y, z;
};

inline CUDA_HOST_DEVICE Point& operator+=(Point &p, const Vec &v) {
	p.x += v.x; p.y += v.y; p.z += v.z; return p;
}

inline CUDA_HOST_DEVICE Point& operator-=(Point &p, const Vec &v) {
	p.x -= v.x; p.y -= v.y; p.z -= v.z; return p;
}

inline CUDA_HOST_DEVICE Point operator+(const Point &p, const Vec &v) {
	return Point(p.x + v.x, p.y + v.y, p.z + v.z);
}
inline CUDA_HOST_DEVICE Point operator+(const Point &p, float d) {
	return Point(p.x + d, p.y + d, p.z + d);
}

inline CUDA_HOST_DEVICE Vec operator-(const Point &p1, const Point &p2) {
	return Vec(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
}
inline CUDA_HOST_DEVICE Point operator-(const Point &p, float d) {
	return Point(p.x - d, p.y - d, p.z - d);
}

inline CUDA_HOST_DEVICE float Point::Distance(const Point &p) const {
	return (*this - p).Length();
}
inline CUDA_HOST_DEVICE float Point::DistanceSquared(const Point &p) const {
	return (*this - p).LengthSquared();
}

/**
 * Represents a 4x4 matrix
 */
struct Matrix {
    Matrix() {
        m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.f;
        m[0][1] = m[0][2] = m[0][3] = m[1][0] = 
		m[1][2] = m[1][3] = m[2][0] = m[2][1] = 
		m[2][3] = m[3][0] = m[3][1] = m[3][2] = 0.f;
    }
    Matrix(float mat[4][4]) {
		memcpy(m, mat, 16*sizeof(float));
	}
    Matrix(const Vec &v0, const Vec &v1, const Vec &v2, const Vec &v3) {
		m[0][0] = v0.x; m[0][1] = v0.y; m[0][2] = v0.z; m[0][3] = v0.w;
		m[1][0] = v1.x; m[1][1] = v1.y; m[1][2] = v1.z; m[1][3] = v1.w;
		m[2][0] = v2.x; m[2][1] = v2.y; m[2][2] = v2.z; m[2][3] = v2.w;
		m[3][0] = v3.x; m[3][1] = v3.y; m[3][2] = v3.z; m[3][3] = v3.w;
	}
    Matrix(float v00, float v01, float v02, float v03,
			float v10, float v11, float v12, float v13,
			float v20, float v21, float v22, float v23,
			float v30, float v31, float v32, float v33) {
		m[0][0] = v00; m[0][1] = v01; m[0][2] = v02; m[0][3] = v03;
		m[1][0] = v10; m[1][1] = v11; m[1][2] = v12; m[1][3] = v13;
		m[2][0] = v20; m[2][1] = v21; m[2][2] = v22; m[2][3] = v23;
		m[3][0] = v30; m[3][1] = v31; m[3][2] = v32; m[3][3] = v33;
	}
    bool operator==(const Matrix &m2) const {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                if (m[i][j] != m2.m[i][j]) return false;
        return true;
    }
    bool operator!=(const Matrix &m2) const {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                if (m[i][j] != m2.m[i][j]) return true;
        return false;
    }
    Matrix Transpose() const {
		return Matrix(m[0][0], m[1][0], m[2][0], m[3][0],
			m[0][1], m[1][1], m[2][1], m[3][1],
			m[0][2], m[1][2], m[2][2], m[3][2],
			m[0][3], m[1][3], m[2][3], m[3][3]);
	}
    Matrix operator*(const Matrix &mat) const {
        Matrix r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                r.m[i][j] = m[i][0] * mat.m[0][j] +
                            m[i][1] * mat.m[1][j] +
                            m[i][2] * mat.m[2][j] +
                            m[i][3] * mat.m[3][j];
        return r;
    }
    Matrix Inverse() const {
		int indxc[4], indxr[4];
		int ipiv[4] = { 0, 0, 0, 0 };
		float minv[4][4];
		memcpy(minv, m, 4*4*sizeof(float));

		for (int i = 0; i < 4; i++) {
			int irow = -1, icol = -1;
			float big = 0.;

			// Choose pivot
			for (int j = 0; j < 4; j++) {
				if (ipiv[j] != 1) {
					for (int k = 0; k < 4; k++) {
						if (ipiv[k] == 0) {
							if (fabsf(minv[j][k]) >= big) {
								big = float(fabsf(minv[j][k]));
								irow = j;
								icol = k;
							}
						}
						else if (ipiv[k] > 1) {
							printf("Singular matrix occured!");
							exit(-1);
						}
					}
				}
			}
			++ipiv[icol];
			// Swap rows _irow_ and _icol_ for pivot
			if (irow != icol) {
				for (int k = 0; k < 4; ++k)
					std::swap(minv[irow][k], minv[icol][k]);
			}
			indxr[i] = irow;
			indxc[i] = icol;
			if (minv[icol][icol] == 0.) {
				printf("Singular matrix occured!");
				exit(-1);
			}

			// Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
			float pivinv = 1.f / minv[icol][icol];
			minv[icol][icol] = 1.f;
			for (int j = 0; j < 4; j++)
				minv[icol][j] *= pivinv;

			// Subtract this row from others to zero out their columns
			for (int j = 0; j < 4; j++) {
				if (j != icol) {
					float save = minv[j][icol];
					minv[j][icol] = 0;
					for (int k = 0; k < 4; k++)
						minv[j][k] -= minv[icol][k]*save;
				}
			}
		}
		// Swap columns to reflect permutation
		for (int j = 3; j >= 0; j--) {
			if (indxr[j] != indxc[j]) {
				for (int k = 0; k < 4; k++)
					std::swap(minv[k][indxr[j]], minv[k][indxc[j]]);
			}
		}
		return Matrix(minv);
	}
	Vec GetRow(int row) const {
		assert(row >= 0 && row <= 3);
		return Vec(m[row][0], m[row][1], m[row][2], m[row][3]);
	}
	Vec GetCol(int col) const {
		assert(row >= 0 && row <= 3);
		return Vec(m[0][col], m[1][col], m[2][col], m[3][col]);
	}

    float m[4][4];
};

/**
 * Represents a ray with origin and direction
 */
struct Ray {
	CUDA_HOST_DEVICE Ray() : o(Point()), d(Vec()) {}
	CUDA_HOST_DEVICE Ray(const Point &o_, const Vec &d_) : o(o_), d(d_) {}
	CUDA_HOST_DEVICE Point operator()(float t) const { return o + d*t; }

	Point o;
	Vec d;
};

/**
 * Represents an axis aligned bounding box
 */
struct BBox {
	CUDA_HOST_DEVICE BBox() {
		pMin = Point(INF, INF, INF);
		pMax = Point(-INF, -INF, -INF);
	}
	CUDA_HOST_DEVICE BBox(const Point &p) : pMin(p), pMax(p) { }
	CUDA_HOST_DEVICE BBox(const Point &p1, const Point &p2) {
		pMin = Point(min(p1.x, p2.x), min(p1.y, p2.y), min(p1.z, p2.z));
		pMax = Point(max(p1.x, p2.x), max(p1.y, p2.y), max(p1.z, p2.z));
	}
	CUDA_HOST_DEVICE BBox Union(const Point &p) {
		pMin.x = min(pMin.x, p.x);
		pMin.y = min(pMin.y, p.y);
		pMin.z = min(pMin.z, p.z);
		pMax.x = max(pMax.x, p.x);
		pMax.y = max(pMax.y, p.y);
		pMax.z = max(pMax.z, p.z);
		return *this;
	}
	CUDA_HOST_DEVICE BBox Union(const BBox &b2) {
		pMin.x = min(pMin.x, b2.pMin.x);
		pMin.y = min(pMin.y, b2.pMin.y);
		pMin.z = min(pMin.z, b2.pMin.z);
		pMax.x = max(pMax.x, b2.pMax.x);
		pMax.y = max(pMax.y, b2.pMax.y);
		pMax.z = max(pMax.z, b2.pMax.z);
		return *this;
	}
	CUDA_HOST_DEVICE bool Overlaps(const BBox &b) const {
		bool x = (pMax.x >= b.pMin.x) && (pMin.x <= b.pMax.x);
		bool y = (pMax.y >= b.pMin.y) && (pMin.y <= b.pMax.y);
		bool z = (pMax.z >= b.pMin.z) && (pMin.z <= b.pMax.z);
		return (x && y && z);
	}
	CUDA_HOST_DEVICE bool Inside(const Point &pt) const {
		return (pt.x >= pMin.x && pt.x <= pMax.x &&
			pt.y >= pMin.y && pt.y <= pMax.y &&
			pt.z >= pMin.z && pt.z <= pMax.z);
	}
	CUDA_HOST_DEVICE void Expand(float delta) {
		pMin -= Vec(delta, delta, delta);
		pMax += Vec(delta, delta, delta);
	}
	CUDA_HOST_DEVICE float SurfaceArea() const {
		Vec d = pMax - pMin;
		return 2.f * (d.x * d.y + d.x * d.z + d.y * d.z);
	}
	CUDA_HOST_DEVICE float Volume() const {
		Vec d = pMax - pMin;
		return d.x * d.y * d.z;
	}
	CUDA_HOST_DEVICE int MaximumExtent() const {
		Vec diag = pMax - pMin;
		if (diag.x > diag.y && diag.x > diag.z)
			return 0;
		else if (diag.y > diag.z)
			return 1;
		else
			return 2;
	}
	//CUDA_HOST_DEVICE const Point &operator[](int i) const;
	//CUDA_HOST_DEVICE Point &operator[](int i);
	// CUDA_HOST_DEVICE Point Lerp(float tx, float ty, float tz) const {
	// 	return Point(::Lerp(tx, pMin.x, pMax.x), ::Lerp(ty, pMin.y, pMax.y),
	// 		::Lerp(tz, pMin.z, pMax.z));
	// }
	CUDA_HOST_DEVICE Vec Offset(const Point &p) const {
		return Vec((p - pMin) / (pMax - pMin));
	}
	CUDA_HOST_DEVICE bool IntersectP(const Ray &ray, float *hitt0 = NULL,
			float *hitt1 = NULL) const {
		float t0 = EPSILON, t1 = INF;
		for (int i = 0; i < 3; ++i) {
			// Update interval for _i_th bounding box slab
			float invRayDir = 1.f / ray.d[i];
			float tNear = (pMin[i] - ray.o[i]) * invRayDir;
			float tFar  = (pMax[i] - ray.o[i]) * invRayDir;

			// Update parametric interval from slab intersection $t$s
			if (tNear > tFar) {
				float tmp = tFar;
				tFar = tNear;
				tNear = tmp;
				//std::swap(tNear, tFar);
			}
			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar  < t1 ? tFar  : t1;
			if (t0 > t1) return false;
		}
		if (hitt0) *hitt0 = t0;
		if (hitt1) *hitt1 = t1;
		return true;
	}
	CUDA_HOST_DEVICE bool operator==(const BBox &b) const {
		return b.pMin == pMin && b.pMax == pMax;
	}
	CUDA_HOST_DEVICE bool operator!=(const BBox &b) const {
		return b.pMin != pMin || b.pMax != pMax;
	}

	Point pMin, pMax;
};

/**
 * Represents a RGB color value
 */
struct Color {
	inline CUDA_HOST_DEVICE Color() { r = g = b = 0.f; }
	inline CUDA_HOST_DEVICE Color(float r_, float g_=0.f, float b_=0.f) 
		: r(r_), g(g_), b(b_) { }
	inline CUDA_HOST_DEVICE Color(const Color &c) : r(c.r), g(c.g), b(c.b) { }

	inline CUDA_HOST_DEVICE Color operator=(float f) { return Color(f, f, f); }

	inline CUDA_HOST_DEVICE float operator[](int i) const {
		assert(i >= 0 && i <= 2); return (&r)[i];
	}
	inline CUDA_HOST_DEVICE float& operator[](int i) {
		assert(i >= 0 && i <= 2); return (&r)[i];
	}

	inline CUDA_HOST_DEVICE bool IsBlack() const { return r == 0.f && g == 0.f && b == 0.f; }
	inline CUDA_HOST_DEVICE float Max() const { return max(r, max(g,b)); }
	inline CUDA_HOST_DEVICE float Y() const { return r*0.2126f + g*0.7152f + b*0.0722f; }

	float r, g, b;
};

inline CUDA_HOST_DEVICE Color& operator+=(Color &c1, const Color &c2) {
	c1.r += c2.r; c1.g += c2.g; c1.b += c2.b; return c1;
}
inline CUDA_HOST_DEVICE Color& operator+=(Color &c, float d) {
	c.r += d; c.g += d; c.b += d; return c;
}

inline CUDA_HOST_DEVICE Color& operator-=(Color &c1, const Color &c2) {
	c1.r -= c2.r; c1.g -= c2.g; c1.b -= c2.b; return c1;
}
inline CUDA_HOST_DEVICE Color& operator-=(Color &c, float d) {
	c.r -= d; c.g -= d; c.b -= d; return c;
}

inline CUDA_HOST_DEVICE Color& operator*=(Color &c1, const Color &c2) {
	c1.r *= c2.r; c1.g *= c2.g; c1.b *= c2.b; return c1;
}
inline CUDA_HOST_DEVICE Color& operator*=(Color &c, float d) {
	c.r *= d; c.g *= d; c.b *= d; return c;
}

inline CUDA_HOST_DEVICE Color& operator/=(Color &c1, const Color &c2) {
	c1.r /= c2.r; c1.g /= c2.g; c1.b /= c2.b; return c1;
}
inline CUDA_HOST_DEVICE Color& operator/=(Color &c, float d) {
	assert(d != 0.f); const float inv = 1.f / d; return c *= inv;
}

inline CUDA_HOST_DEVICE Color operator+(const Color &c1, const Color &c2) {
	return Color(c1.r + c2.r, c1.g + c2.g, c1.b + c2.b);
}
inline CUDA_HOST_DEVICE Color operator+(const Color &c, float d) {
	return Color(c.r + d, c.g + d, c.b + d);
}
inline CUDA_HOST_DEVICE Color operator+(float d, const Color &c) {
	return Color(c.r + d, c.g + d, c.b + d);
}

inline CUDA_HOST_DEVICE Color operator-(const Color &c1, const Color &c2) {
	return Color(c1.r - c2.r, c1.g - c2.g, c1.b - c2.b);
}
inline CUDA_HOST_DEVICE Color operator-(const Color &c, float d) {
	return Color(c.r - d, c.g - d, c.b - d);
}
inline CUDA_HOST_DEVICE Color operator-(float d, const Color &c) {
	return Color(c.r - d, c.g - d, c.b - d);
}

inline CUDA_HOST_DEVICE Color operator*(const Color &c1, const Color &c2) {
	return Color(c1.r*c2.r, c1.g*c2.g, c1.b*c2.b);
}
inline CUDA_HOST_DEVICE Color operator*(const Color &c, float d) {
	return Color(c.r*d, c.g*d, c.b*d);
}
inline CUDA_HOST_DEVICE Color operator*(float d, const Color &c) {
	return Color(c.r*d, c.g*d, c.b*d);
}

inline CUDA_HOST_DEVICE Color operator/(const Color &c1, const Color &c2) {
	return Color(c1.r / c2.r, c1.g / c2.g, c1.b / c2.b);
}
inline CUDA_HOST_DEVICE Color operator/(const Color &c, float d) {
	assert(d != 0.f); const float inv = 1.f / d; return c*inv;
}
inline CUDA_HOST_DEVICE Color operator/(float d, const Color &c) {
	assert(d != 0.f); const float inv = 1.f / d; return c*inv;
}

inline CUDA_DEVICE void CoordinateSystem(const Vec &v1, Vec *v2, Vec *v3) {
    if (fabsf(v1.x) > fabsf(v1.y)) {
        float invLen = 1.f / sqrtf(v1.x*v1.x + v1.z*v1.z);
        *v2 = Vec(-v1.z * invLen, 0.f, v1.x * invLen);
    }
    else {
        float invLen = 1.f / sqrtf(v1.y*v1.y + v1.z*v1.z);
        *v2 = Vec(0.f, v1.z * invLen, -v1.y * invLen);
    }
    *v3 = v1.Cross(*v2);
}

#endif /* __VECMATH_H__ */