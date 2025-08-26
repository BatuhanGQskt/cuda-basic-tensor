#ifndef PARTICLE_H
#define PARTICLE_H

#include "point3d.h"

/**
 * @brief A particle with physics properties
 */
struct Particle {
    Point3D position;
    Point3D velocity;
    Point3D acceleration;
    float mass;
    float charge;
    bool isActive;
    
    // Constructors
    __host__ __device__ Particle();
    __host__ __device__ Particle(const Point3D& pos, const Point3D& vel, float m, float q = 1.0f);
    
    // Physics simulation methods
    __device__ void updatePosition(float deltaTime);
    __device__ void updateVelocity(float deltaTime);
    __device__ void applyForce(const Point3D& force, float deltaTime);
    __device__ void integrate(float deltaTime);
    
    // Property calculations
    __host__ __device__ float kineticEnergy() const;
    __host__ __device__ float momentum() const;
    __host__ __device__ Point3D momentumVector() const;
    
    // Interaction methods
    __device__ void applyGravitationalForce(const Particle& other, float G = 1.0f);
    __device__ void applyElectrostaticForce(const Particle& other, float k = 1.0f);
    __device__ bool checkCollision(const Particle& other, float radius = 1.0f) const;
    
    // State management
    __device__ void reset();
    __device__ void setActive(bool active);
};

// CUDA kernel declarations
__global__ void updateParticles(Particle* particles, int N, float deltaTime);
__global__ void applyGravity(Particle* particles, int N, float G);
__global__ void detectCollisions(Particle* particles, int* collisions, int N, float radius);

// Host utility functions
__host__ void initializeRandomParticles(Particle* particles, int N, float posRange, float velRange);
__host__ void printParticleStats(const Particle* particles, int N);

#endif // PARTICLE_H
