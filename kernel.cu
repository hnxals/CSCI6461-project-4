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

void CUDAMain() {

}


int main() {
	CLoopMain();
	CUDAMain();

	return 0;
}

