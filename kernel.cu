#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void CLoopMatrixMultiplication(int* M, int* N, int* P, int Width) {
	for (int i = 0; i < Width; ++i)
		for (int j = 0; j < Width; ++j) {
			int sum = 0;
			for (int k = 0; k < Width; ++k) {
				int a = M[i * Width + k];
				int b = N[k * Width + j];
				sum += a * b;
				
			}
			P[i * Width + j] = sum;
		}
}

void PrintResult(int* p) {
	int count = 0;
	for (int i = 0; i < 256; i++)
	{
		printf("%d ", p[i]);
		count++;
		if (count == 16) {
			count = 0;
			printf("\n");
		}
	}
}

void CLoopMain() {
	int m[256], n[256], p[256];
	for (int i = 0; i < 256; i++)
	{
		m[i] = 1;
		n[i] = 2;
		p[i] = 0;
	}

	CLoopMatrixMultiplication(m, n, p, 16);

	printf("C loop result:\n");
	PrintResult(p);
}

__global__ void MatrixMutilplicationKernel(int* Md, int* Nd, int* Pd, int width) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Pvalue = 0;

	for (int k = 0; k < width; ++k)
	{
		int Mdelement = Md[ty * width + k];
		int Ndelement = Nd[k * width + tx];
		Pvalue += Mdelement * Ndelement;
	}

	Pd[ty * width + tx] = Pvalue;
}

void CUDAMatrixMutilplication(int* m, int* n, int* p, int width) {
	int size = width * width * sizeof(int);
	int* Md; 
	int* Nd; 
	int* Pd;
	cudaMalloc((void**)&Md, size);
	cudaMemcpy(Md, m, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Nd, size);
	cudaMemcpy(Nd, n, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Pd, size);

	dim3 dimBlock(width, width);
	dim3 dimGrid(1, 1);
	MatrixMutilplicationKernel <<<dimGrid, dimBlock>>> (Md, Nd, Pd, width);

	cudaMemcpy(p, Pd, size, cudaMemcpyDeviceToHost);
	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);
}

void CUDAMain() {
	int m[256], n[256], p[256];
	for (int i = 0; i < 256; i++)
	{
		m[i] = 1;
		n[i] = 2;
		p[i] = 0;
	}

	CUDAMatrixMutilplication(m, n, p, 16);
	printf("CUDA result:\n");
	PrintResult(p);
}


int main() {
	CLoopMain();
	CUDAMain();

	return 0;
}

