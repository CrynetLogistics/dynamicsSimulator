#include <iostream>
#include <ctime>
#include "SDL.h"
#include "stdio.h"
#include "math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structures.h"
#include "stdlib.h"
#include "PointMass.h"

#undef main
#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720
#define DISPLAY_TIME 30000

#define NUM_OF_BLOCKS 900

////////////////ATOMIC THINGS ONLY PROVIDE SYNCHRONISATION BETWEEN THREADS IN THE SAME BLOCK
#define SIMULATION_THREADS_PER_BLOCK 1024

//#define NUM_OF_POINTS 10240
#define NUM_OF_POINTS 12000
#define EPSILON 0.5
#define TIME_INTERVAL 0.05
#define ITERATIONS_TO_CALCULATE 2000

#define INITIAL_POSITION_X rand()%2560-1280
#define INITIAL_POSITION_Y rand()%1440-720
#define INITIAL_POSITION_Z 0
#define INITIAL_VELOCITY_X (float)(rand()%200-100)/10
#define INITIAL_VELOCITY_Y (float)(rand()%200-100)/10
#define INITIAL_VELOCITY_Z (float)(rand()%200-100)/10
#define INITIAL_ACCELERATION_X (float)(rand()%20-10)/10
#define INITIAL_ACCELERATION_Y (float)(rand()%20-10)/10
#define INITIAL_ACCELERATION_Z (float)(rand()%20-10)/10
#define INITIAL_MASS (float)(rand()%10)/1000

point_t* initialiseColourSystem(int x, int y);
PointMass* initialiseSystem(SDL_Renderer* renderer, int x, int y);
void iterateSystem(SDL_Renderer* renderer, PointMass* d_dynamicSystem, point_t* d_colourGrid, int x, int y);
void destroySystem(PointMass* d_dynamicSystem, point_t* d_colourGrid);

int main()
{
    SDL_Window* window = NULL;
	SDL_Init(SDL_INIT_EVERYTHING);

	//create window
	window = SDL_CreateWindow("Force Field Simulation", 
		SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    
	SDL_Renderer *renderer = NULL;
	renderer = SDL_CreateRenderer(window, 0, SDL_RENDERER_ACCELERATED);
	//BACKGROUND COLOUR SET
	SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
	SDL_RenderClear(renderer);


	

	PointMass* d_dynamicSystem;
	point_t* d_colourGrid;
	d_colourGrid = initialiseColourSystem(SCREEN_WIDTH, SCREEN_HEIGHT);
	d_dynamicSystem = initialiseSystem(renderer, SCREEN_WIDTH, SCREEN_HEIGHT);
	SDL_RenderPresent(renderer);


	for(int i=0;i<ITERATIONS_TO_CALCULATE;i++){
		

		iterateSystem(renderer, d_dynamicSystem, d_colourGrid, SCREEN_WIDTH, SCREEN_HEIGHT);
		SDL_RenderPresent(renderer);
	}




	printf("done");
	destroySystem(d_dynamicSystem, d_colourGrid);
	SDL_Delay(DISPLAY_TIME);
	//Destroy window
    SDL_DestroyWindow(window);
    //Quit SDL subsystems
    SDL_Quit();
    return 0;
}


__global__ void simulateIteration(PointMass* d_dynamicSystem){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	d_dynamicSystem[index].advanceSimulation(TIME_INTERVAL, d_dynamicSystem);
}


__global__ void pollColours(point_t* d_colourGrid, PointMass* d_dynamicSystem){
	int index = blockIdx.x*blockDim.x+threadIdx.x;

	int xi = (int)d_dynamicSystem[index].getPosition().xt + SCREEN_WIDTH/2;
	int yi = (int)d_dynamicSystem[index].getPosition().yt + SCREEN_HEIGHT/2;

	//int colourIndex = yi * SCREEN_WIDTH + xi;
	__syncthreads();
	if(xi<SCREEN_WIDTH && yi<SCREEN_HEIGHT && xi>0 && yi>0){
		d_colourGrid[index].x = xi;
		d_colourGrid[index].y = yi;
		d_colourGrid[index].r = 255;
		d_colourGrid[index].g = 255;
		d_colourGrid[index].b = 255;
	}
	//__syncthreads();
	//printf("%i ---- %i\n", *colIdx, *colIdx);

}

point_t* initialiseColourSystem(int x, int y){
	point_t* h_colourGrid = (point_t*)calloc(SCREEN_HEIGHT*SCREEN_WIDTH, sizeof(point_t));
	point_t* d_colourGrid;
	cudaMalloc((void**) &d_colourGrid, sizeof(point_t)*SCREEN_HEIGHT*SCREEN_WIDTH);
	cudaMemcpy(d_colourGrid, h_colourGrid, sizeof(point_t)*SCREEN_HEIGHT*SCREEN_WIDTH, cudaMemcpyHostToDevice);
	free(h_colourGrid);
	return d_colourGrid;
}

PointMass* initialiseSystem(SDL_Renderer* renderer, int x, int y){

	srand(time(NULL));

	PointMass* h_dynamicSystem = (PointMass*)malloc(NUM_OF_POINTS*sizeof(PointMass));

	for(int i=0;i<NUM_OF_POINTS-1000;i++){
		h_dynamicSystem[i] = PointMass(INITIAL_POSITION_X,
									   INITIAL_POSITION_Y,
									   INITIAL_POSITION_Z,
									   INITIAL_VELOCITY_X,
									   INITIAL_VELOCITY_Y,
									   INITIAL_VELOCITY_Z,
									   INITIAL_ACCELERATION_X,
									   INITIAL_ACCELERATION_Y,
									   INITIAL_ACCELERATION_Z,
									   INITIAL_MASS);
	}

	for(int i=NUM_OF_POINTS-1000;i<NUM_OF_POINTS;i++){
		h_dynamicSystem[i] = PointMass((INITIAL_POSITION_X)/100,
									   (INITIAL_POSITION_Y)/100,
									   (INITIAL_POSITION_Z)/100,
									   INITIAL_VELOCITY_X,
									   INITIAL_VELOCITY_Y,
									   INITIAL_VELOCITY_Z,
									   INITIAL_ACCELERATION_X,
									   INITIAL_ACCELERATION_Y,
									   INITIAL_ACCELERATION_Z,
									   INITIAL_MASS);
	}
	
	PointMass* d_dynamicSystem;

	cudaMalloc((void**) &d_dynamicSystem, NUM_OF_POINTS*sizeof(PointMass));
	cudaMemcpy(d_dynamicSystem, h_dynamicSystem, NUM_OF_POINTS*sizeof(PointMass), cudaMemcpyHostToDevice);

	return d_dynamicSystem;
}

void iterateSystem(SDL_Renderer* renderer, PointMass* d_dynamicSystem, point_t* d_colourGrid, int x, int y){

	point_t* h_colourGrid = (point_t*)malloc(sizeof(point_t)*SCREEN_HEIGHT*SCREEN_WIDTH);

	simulateIteration<<<NUM_OF_POINTS/SIMULATION_THREADS_PER_BLOCK,SIMULATION_THREADS_PER_BLOCK>>>(d_dynamicSystem);

	cudaMemset(d_colourGrid, 0, sizeof(point_t)*SCREEN_HEIGHT*SCREEN_WIDTH);

	pollColours<<<NUM_OF_POINTS/SIMULATION_THREADS_PER_BLOCK,SIMULATION_THREADS_PER_BLOCK>>>(d_colourGrid, d_dynamicSystem);

	cudaMemcpy(h_colourGrid, d_colourGrid, sizeof(point_t)*SCREEN_HEIGHT*SCREEN_WIDTH, cudaMemcpyDeviceToHost);
	//END OF GPU CALLING CUDA CODE




	SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
	SDL_RenderClear(renderer);
	SDL_RenderPresent(renderer);

	for(int i=0;i<NUM_OF_POINTS;i++){
		if(h_colourGrid[i].r<=255 && h_colourGrid[i].g<=255 && h_colourGrid[i].b<=255){
			SDL_SetRenderDrawColor(renderer, h_colourGrid[i].r, h_colourGrid[i].g, h_colourGrid[i].b, 255);
		}else{
			//draw bright flourescent pink for regions out of colour range nice one zl
			SDL_SetRenderDrawColor(renderer, 255, 0, 255, 255);
		}

		SDL_RenderDrawPoint(renderer, h_colourGrid[i].x, h_colourGrid[i].y);
	}

	free(h_colourGrid);
}

void destroySystem(PointMass* d_dynamicSystem, point_t* d_colourGrid){
	cudaFree(d_dynamicSystem);
	cudaFree(d_colourGrid);
}