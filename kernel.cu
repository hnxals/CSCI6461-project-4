#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <Windows.h>

void CLoopMatrixMultiplication(float* M, float* N, float* P, int Width) {
	for (int i = 0; i < Width; ++i) {
		for (int j = 0; j < Width; ++j) {
			float sum = 0;
			for (int k = 0; k < Width; ++k) {
				float a = M[i * Width + k];
				float b = N[k * Width + j];
				sum += a * b;
			}
			P[i * Width + j] = sum;
		}
	}
}

void PrintResult(float* p) {
	int count = 0;
	for (int i = 0; i < 256; i++)
	{
		printf("%.lf ", p[i]);
		count++;
		if (count == 16) {
			count = 0;
			printf("\n");
		}
	}
}

void CLoopMain() {
	float *m, *n, *p;
	int size = 256 * sizeof(float);
	m = (float*)malloc(size);
	n = (float*)malloc(size);
	p = (float*)malloc(size);
	for (int i = 0; i < 256; i++)
	{
		m[i] = 1;
		n[i] = 2;
		p[i] = 0;
	}
	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

	CLoopMatrixMultiplication(m, n, p, 16);

	QueryPerformanceCounter(&t2);
	printf("C loop run time:%f \n C loop result:\n", (t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart);
	PrintResult(p);
	free(m);
	free(n);
	free(p);
}

__global__ void MatrixMutilplicationKernel(float* Md, float* Nd, float* Pd, int width) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float Pvalue = 0;
	for (int k = 0; k < width; ++k)
	{
		float Mdelement = Md[ty * width + k];
		float Ndelement = Nd[k * width + tx];
		Pvalue += Mdelement * Ndelement;
	}
	Pd[ty * width + tx] = Pvalue;
}

void CUDAMatrixMutilplication(float* m, float* n, float* p, int width) {
	int size = width * width * sizeof(float);
	float* Md;
	float* Nd;
	float* Pd;
	cudaMalloc((void**)&Md, size);
	cudaMemcpy(Md, m, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Nd, size);
	cudaMemcpy(Nd, n, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Pd, size);

	dim3 dimBlock(width, width);
	dim3 dimGrid(1, 1);

	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

	MatrixMutilplicationKernel <<<dimGrid, dimBlock>>> (Md, Nd, Pd, width);

	QueryPerformanceCounter(&t2);
	printf("CUDA run time:%f \n CUDA result:\n", (t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart);

	cudaMemcpy(p, Pd, size, cudaMemcpyDeviceToHost);
	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);
}

void CUDAMain() {
	float* m, * n, * p;
	int size = 256 * sizeof(float);
	m = (float*)malloc(size);
	n = (float*)malloc(size);
	p = (float*)malloc(size);
	for (int i = 0; i < 256; i++)
	{
		m[i] = 1;
		n[i] = 2;
		p[i] = 0;
	}
	CUDAMatrixMutilplication(m, n, p, 16);
	PrintResult(p);
	free(m);
	free(n);
	free(p);
}


int main() {
	CLoopMain();
	CUDAMain();
	return 0;
}

