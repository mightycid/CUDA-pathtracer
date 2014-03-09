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

struct Vec {
	inline CUDA_HOST_DEVICE Vec() { x = y = z = 0.f; }
	inline CUDA_HOST_DEVICE Vec(float x_, float y_=0.f, float z_=0.f) 
		: x(x_), y(y_), z(z_) { assert(!HasNaNs()); }
	inline CUDA_HOST_DEVICE Vec(const Vec &v) : x(v.x), y(v.y), z(v.z) { 
		assert(!HasNaNs()); 
	}

	inline CUDA_HOST_DEVICE bool HasNaNs() const { 
#ifndef __CUDA_ARCH__
		return isnan(x) || isnan(y) || isnan(z);
#else
		return true;
#endif
	}

	inline CUDA_HOST_DEVICE float operator[](int i) const {
		assert(i >= 0 && i <= 2); return (&x)[i];
	}
	inline CUDA_HOST_DEVICE float& operator[](int i) {
		assert(i >= 0 && i <= 2); return (&x)[i];
	}

	inline CUDA_HOST_DEVICE Vec operator-() const {
		return Vec(-x, -y, -z);
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
	inline CUDA_HOST_DEVICE Vec Normalize() const;
	inline CUDA_HOST_DEVICE Vec& Normalize();
	inline CUDA_HOST_DEVICE Vec& Normalize(float *l);

	float x, y, z;
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

inline CUDA_HOST_DEVICE Vec Vec::Normalize() const {
	return *this / Length();
}
inline CUDA_HOST_DEVICE Vec& Vec::Normalize() {
	return *this = *this / Length();
}
inline CUDA_HOST_DEVICE Vec& Vec::Normalize(float *l) {
	*l = Length(); return *this = *this / *l;
}

struct Point {
	inline CUDA_HOST_DEVICE Point() { x = y = z = 0.f; assert(!HasNaNs()); }
	inline CUDA_HOST_DEVICE Point(float x_, float y_=0.f, float z_=0.f)
		: x(x_), y(y_), z(z_) { assert(!HasNaNs()); }
	inline CUDA_HOST_DEVICE Point(const Point &p) : x(p.x), y(p.y), z(p.z) {
		assert(!HasNaNs());
	}

	inline CUDA_HOST_DEVICE bool HasNaNs() const {
#ifndef __CUDA_ARCH__
		return isnan(x) || isnan(y) || isnan(z);
#else
		return true;
#endif
	}

	inline CUDA_HOST_DEVICE Point& operator=(const Point &p) {
		x = p.x; y = p.y; z = p.z; return *this;
	}
	inline CUDA_HOST_DEVICE float operator[](int i) const {
		assert(i >= 0 && i <= 2); return (&x)[i];
	}
	inline CUDA_HOST_DEVICE float& operator[](int i) {
		assert(i >= 0 && i <= 2); return (&x)[i];
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
// inline CUDA_HOST_DEVICE Point& operator+=(Point &p, float d) {
// 	v.x += d; v.y += d; v.z += d; return v;
// }

inline CUDA_HOST_DEVICE Point& operator-=(Point &p, const Vec &v) {
	p.x -= v.x; p.y -= v.y; p.z -= v.z; return p;
}
// inline CUDA_HOST_DEVICE Point& operator-=(Point &p, float d) {
// 	v.x -= d; v.y -= d; v.z -= d; return v;
// }

// inline CUDA_HOST_DEVICE Point& operator*=(Point &p1, const Point &p2) {
// 	p1.x *= p2.x; p1.y *= p2.y; p1.z *= p2.z; return p1; }
// inline CUDA_HOST_DEVICE Point& operator*=(Point &p, float d) {
// 	v.x *= d; v.y *= d; v.z *= d; return v;
// }

// inline CUDA_HOST_DEVICE Point& operator/=(Point &p1, const Point &p2) {
// 	p1.x /= p2.x; p1.y /= p2.y; p1.z /= p2.z; return p1;
// }
// inline CUDA_HOST_DEVICE Point& operator/=(Point &p, float d) {
// 	assert(d != 0.f); const float inv = 1.f / d; return v *= inv;
// }

inline CUDA_HOST_DEVICE Point operator+(const Point &p, const Vec &v) {
	return Point(p.x + v.x, p.y + v.y, p.z + v.z);
}
// inline CUDA_HOST_DEVICE Point operator+(const Point &p, float d) {
// 	return Point(v.x + d, v.y + d, v.z + d);
// }
// inline CUDA_HOST_DEVICE Point operator+(float d, const Point &p) {
// 	return Point(v.x + d, v.y + d, v.z + d);
// }

inline CUDA_HOST_DEVICE Vec operator-(const Point &p1, const Point &p2) {
	return Vec(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
}
// inline CUDA_HOST_DEVICE Point operator-(const Point &p, float d) {
// 	return Point(v.x - d, v.y - d, v.z - d);
// }
// inline CUDA_HOST_DEVICE Point operator-(float d, const Point &p) {
// 	return Point(v.x - d, v.y - d, v.z - d);
// }

// inline CUDA_HOST_DEVICE Point operator*(const Point &p1, const Point &p2) {
// 	return Point(p1.x*p2.x, p1.y*p2.y, p1.z*p2.z);
// }
// inline CUDA_HOST_DEVICE Point operator*(const Point &p, float d) {
// 	return Point(v.x*d, v.y*d, v.z*d);
// }
// inline CUDA_HOST_DEVICE Point operator*(float d, const Point &p) {
// 	return Point(v.x*d, v.y*d, v.z*d);
// }

// inline CUDA_HOST_DEVICE Point operator/(const Point &p1, const Point &p2) {
// 	return Point(p1.x / p2.x, p1.y / p2.y, p1.z / p2.z);
// }
// inline CUDA_HOST_DEVICE Point operator/(const Point &p, float d) {
// 	assert(d != 0.f); const float inv = 1.f / d; return v*inv;
// }
// inline CUDA_HOST_DEVICE Point operator/(float d, const Point &p) {
// 	assert(d != 0.f); const float inv = 1.f / d; return v*inv;
// }

inline CUDA_HOST_DEVICE float Point::Distance(const Point &p) const {
	return (*this - p).Length();
}
inline CUDA_HOST_DEVICE float Point::DistanceSquared(const Point &p) const {
	return (*this - p).LengthSquared();
}


struct Ray {
	CUDA_HOST_DEVICE Ray() : o(Point()), d(Vec()) {}
	CUDA_HOST_DEVICE Ray(const Point &o_, const Vec &d_) : o(o_), d(d_) {}
	CUDA_HOST_DEVICE ~Ray() {}
	CUDA_HOST_DEVICE Point operator()(float t) const { assert(!isnan(t)); return o + d*t; }

	Point o;
	Vec d;
};

struct BBox {
	// BBox Public Methods
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
	CUDA_HOST_DEVICE const Point &operator[](int i) const;
	CUDA_HOST_DEVICE Point &operator[](int i);
	// CUDA_HOST_DEVICE Point Lerp(float tx, float ty, float tz) const {
	// 	return Point(::Lerp(tx, pMin.x, pMax.x), ::Lerp(ty, pMin.y, pMax.y),
	// 		::Lerp(tz, pMin.z, pMax.z));
	// }
	CUDA_HOST_DEVICE Vec Offset(const Point &p) const {
		return Vec((p - pMin) / (pMax - pMin));
	}
	CUDA_HOST_DEVICE bool IntersectP(const Ray &ray, float *hitt0 = NULL,
		float *hitt1 = NULL) const;
	CUDA_HOST_DEVICE bool operator==(const BBox &b) const {
		return b.pMin == pMin && b.pMax == pMax;
	}
	CUDA_HOST_DEVICE bool operator!=(const BBox &b) const {
		return b.pMin != pMin || b.pMax != pMax;
	}

	Point pMin, pMax;
};

struct Color {
	inline CUDA_HOST_DEVICE Color() { r = g = b = 0.f; }
	inline CUDA_HOST_DEVICE Color(float r_, float g_=0.f, float b_=0.f) 
		: r(r_), g(g_), b(b_) { assert(!HasNaNs()); }
	inline CUDA_HOST_DEVICE Color(const Color &c) : r(c.r), g(c.g), b(c.b) {
		assert(!HasNaNs());
	}

	inline CUDA_HOST_DEVICE Color operator=(float f) { return Color(f, f, f); }

	inline CUDA_HOST_DEVICE bool HasNaNs() const {
#ifndef __CUDA_ARCH__
		return isnan(r) || isnan(g) || isnan(b);
#else
		return true;
#endif
	}

	inline CUDA_HOST_DEVICE float operator[](int i) const {
		assert(i >= 0 && i <= 2); return (&r)[i];
	}
	inline CUDA_HOST_DEVICE float& operator[](int i) {
		assert(i >= 0 && i <= 2); return (&r)[i];
	}

	inline CUDA_HOST_DEVICE bool IsBlack() const { return r == 0.f && g == 0.f && b == 0.f; }
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
	assert(d != 0.f); const float inc = 1.f / d; return c *= inc;
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

#endif /* __VECMATH_H__ */