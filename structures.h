#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct vertex{
	float x;
	float y;
	float z;
} vertex_t;

typedef struct colour{
	float r;
	float g;
	float b;
} colour_t;

typedef struct point{
	int r;
	int g;
	int b;
	int x;
	int y;
} point_t;