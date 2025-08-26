#ifndef SHAPES_H
#define SHAPES_H

#include "point3d.h"

/**
 * @brief Base class for geometric shapes
 */
class Shape {
protected:
    Point3D center;
    
public:
    __host__ __device__ Shape(const Point3D& center = Point3D());
    __device__ virtual ~Shape() = default;
    
    // Pure virtual functions
    __device__ virtual float area() const = 0;
    __device__ virtual float perimeter() const = 0;
    __device__ virtual bool containsPoint(const Point3D& point) const = 0;
    
    // Common methods
    __device__ void translate(const Point3D& offset);
    __device__ Point3D getCenter() const;
    __device__ void setCenter(const Point3D& newCenter);
};

/**
 * @brief Circle/Sphere shape
 */
class Circle : public Shape {
private:
    float radius;
    
public:
    __host__ __device__ Circle(const Point3D& center = Point3D(), float r = 1.0f);
    
    // Overridden virtual functions
    __device__ float area() const override;
    __device__ float perimeter() const override;
    __device__ bool containsPoint(const Point3D& point) const override;
    
    // Circle-specific methods
    __device__ float getRadius() const;
    __device__ void setRadius(float r);
    __device__ float volume() const; // Sphere volume
    __device__ float surfaceArea() const; // Sphere surface area
};

/**
 * @brief Rectangle shape
 */
class Rectangle : public Shape {
private:
    float width, height;
    
public:
    __host__ __device__ Rectangle(const Point3D& center = Point3D(), float w = 1.0f, float h = 1.0f);
    
    // Overridden virtual functions
    __device__ float area() const override;
    __device__ float perimeter() const override;
    __device__ bool containsPoint(const Point3D& point) const override;
    
    // Rectangle-specific methods
    __device__ float getWidth() const;
    __device__ float getHeight() const;
    __device__ void setDimensions(float w, float h);
    __device__ Point3D getTopLeft() const;
    __device__ Point3D getBottomRight() const;
};

// CUDA kernel declarations
__global__ void processShapes(Shape** shapes, float* areas, float* perimeters, int N);
__global__ void translateShapes(Shape** shapes, Point3D offset, int N);
__global__ void checkPointInShapes(Shape** shapes, Point3D* points, bool* results, int N);

// Factory functions
__host__ Circle* createRandomCircles(int N, float centerRange, float radiusRange);
__host__ Rectangle* createRandomRectangles(int N, float centerRange, float sizeRange);

#endif // SHAPES_H
