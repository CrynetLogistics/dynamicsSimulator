#pragma once
#include "structures.h"
#include "vector_t.h"
class PointMass
{
private:
	vector_t position;
	vector_t velocity;
	vector_t acceleration;
	float mass;

	__host__ __device__ void applyForceField(float timeInterval, PointMass* d_dynamicSystem);
public:
	__host__ __device__ PointMass(float posX, float posY, float posZ, float velX, float velY, float velZ, float accX, float accY, float accZ, float mass);
	__host__ __device__ void advanceSimulation(float timeInterval, PointMass* d_dynamicSystem);
	__host__ __device__ float getMass(void);
	__host__ __device__ vector_t getPosition(void);
	__host__ __device__ ~PointMass(void);
};