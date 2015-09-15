#include "PointMass.h"

#define GRAVITATIONAL_G 1

__host__ __device__ PointMass::PointMass(float posX, float posY, float posZ, float velX, float velY, float velZ, float accX, float accY, float accZ, float mass)
{
	this->mass = mass;
	position.xt = posX;
	position.yt = posY;
	position.zt = posZ;
	//velocity.xt = velX;
	//velocity.yt = velY;

	velocity.xt = posY/100;
	velocity.yt = -posX/100;

	velocity.zt = velZ;
	acceleration.xt = accX;
	acceleration.yt = accY;
	acceleration.zt = accZ;
}

__host__ __device__ void PointMass::applyForceField(float timeInterval, PointMass* d_dynamicSystem){
	////F = ma
	//acceleration.xt += (/*-position.xt/100*/ - position.yt/0.01)/mass;
	//acceleration.yt += (/*-position.yt/100*/ + position.xt/0.01)/mass;
	//acceleration.zt += 0;

	//Impulse, J = Ft = mv - mu
	//velocity.xt += (force.xt*timeInterval)/mass;
	//velocity.yt += (force.yt*timeInterval)/mass;
	//velocity.zt += (force.zt*timeInterval)/mass;

	//velocity.xt += (force.xt*timeInterval)/mass;
	//velocity.yt += (force.yt*timeInterval)/mass;
	//velocity.zt += (force.zt*timeInterval)/mass;

	acceleration.xt = 0;
	acceleration.yt = 0;
	acceleration.zt = 0;

	for(int i=0; i<1024; i++){
		float distance = (position.xt - d_dynamicSystem[i].getPosition().xt)*(position.xt - d_dynamicSystem[i].getPosition().xt)+
						 (position.yt - d_dynamicSystem[i].getPosition().yt)*(position.yt - d_dynamicSystem[i].getPosition().yt)+
						 (position.zt - d_dynamicSystem[i].getPosition().zt)*(position.zt - d_dynamicSystem[i].getPosition().zt);

		//acceleration.xt += (d_dynamicSystem[i].getPosition().xt - position.xt)*GRAVITATIONAL_G*mass*d_dynamicSystem[i].getMass()/distance;
		//acceleration.yt += (d_dynamicSystem[i].getPosition().yt - position.yt)*GRAVITATIONAL_G*mass*d_dynamicSystem[i].getMass()/distance;
		//acceleration.zt += (d_dynamicSystem[i].getPosition().zt - position.zt)*GRAVITATIONAL_G*mass*d_dynamicSystem[i].getMass()/distance;
		if(distance!=0){
			acceleration.xt +=  (d_dynamicSystem[i].getPosition().xt - position.xt)*GRAVITATIONAL_G/distance;
			acceleration.yt +=  (d_dynamicSystem[i].getPosition().yt - position.yt)*GRAVITATIONAL_G/distance;
			acceleration.zt +=  (d_dynamicSystem[i].getPosition().zt - position.zt)*GRAVITATIONAL_G/distance;
		}
	}

	//acceleration.xt = -1*position.xt*(velocity.xt*velocity.xt + velocity.yt*velocity.yt + velocity.zt*velocity.zt)/((position.xt*position.xt + position.yt*position.yt + position.zt*position.zt)*mass);
	//acceleration.yt = -1*position.yt*(velocity.xt*velocity.xt + velocity.yt*velocity.yt + velocity.zt*velocity.zt)/((position.xt*position.xt + position.yt*position.yt + position.zt*position.zt)*mass);
	//acceleration.zt = -1*position.zt*(velocity.xt*velocity.xt + velocity.yt*velocity.yt + velocity.zt*velocity.zt)/((position.xt*position.xt + position.yt*position.yt + position.zt*position.zt)*mass);
}

__host__ __device__ void PointMass::advanceSimulation(float timeInterval, PointMass* d_dynamicSystem){
	applyForceField(timeInterval, d_dynamicSystem);
	velocity.xt += acceleration.xt;
	velocity.yt += acceleration.yt;
	velocity.zt += acceleration.zt;
	position.xt += velocity.xt;
	position.yt += velocity.yt;
	position.zt += velocity.zt;
}

__host__ __device__ vector_t PointMass::getPosition(void){
	return position;
}

__host__ __device__ float PointMass::getMass(void){
	return mass;
}

__host__ __device__ PointMass::~PointMass(void)
{
}
