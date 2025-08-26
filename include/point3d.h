#ifndef POINT3D_H
#define POINT3D_H

#include <cuda_runtime.h>

/**
 * @brief A 3D point structure with CUDA device functions
 */
struct Point3D {
    float x, y, z;
    
    // Constructors
    __host__ __device__ Point3D();
    __host__ __device__ Point3D(float x, float y, float z);
    
    // Device functions for mathematical operations
    __host__ __device__ float distanceFromOrigin() const;
    __host__ __device__ float distanceTo(const Point3D& other) const;
    __host__ __device__ void normalize();
    __host__ __device__ Point3D normalized() const;
    
    // Operator overloads
    __host__ __device__ Point3D operator+(const Point3D& other) const;
    __host__ __device__ Point3D operator-(const Point3D& other) const;
    __host__ __device__ Point3D operator*(float scalar) const;
    __host__ __device__ Point3D& operator+=(const Point3D& other);
    __host__ __device__ Point3D& operator*=(float scalar);
    
    // Utility functions
    __host__ __device__ float dot(const Point3D& other) const;
    __host__ __device__ Point3D cross(const Point3D& other) const;
    __host__ __device__ float magnitude() const;
    __host__ __device__ float magnitudeSquared() const;
};

// Host utility functions
__host__ void printPoint(const Point3D& point);
__host__ Point3D createRandomPoint(float range);

#endif // POINT3D_H
