#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cmath>

#include "AD.cuh"

#if defined (__CUDACC__)
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif


# define N 3

template <size_t NUMVAR, class Precision>
__host__ __device__ void Print(const Dual<NUMVAR, Precision> &a)
{
	//printf("Ez a print lefutott \n");

	printf(" real: %.3f \n", a.real);
	for (int i = 0; i < N; i++)
	{
		printf(" dual[%d]: %.3f \n", i, a.dual[i]);
	}
	printf("\n");
}

__host__ __device__ void TryVariableInitialization()
{
	// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#host
#if defined (__CUDA_ARCH__)
	// Device code here
	printf(" TryVariableInitialization(): Function is called on the device \n");
#else
	// Host code here
	printf(" TryVariableInitialization(): Function is called on the host \n");
#endif

	// Variable initialized with int
	Dual<N, double> a1 = { 2 };
	Dual<N, float>	a2 = { 2 };
	//Dual<N, int>	a3 = { 2 };			// Integerre nem mûködik...
	printf("Dual double (a1)\n");
	Print(a1);
	printf("Dual float (a2) \n");
	Print(a2);
	//printf("Dual int (a3)\n");
	//Print(a3);

	Dual<N, double> b1 = { 1, 0 };
	Dual<N, float>	b2 = { 2, 1 };
	//Dual<N, int>	b3 = { 3, 2 };

	printf("Dual double (b1) \n");
	Print(b1);
	printf("Dual float (b2) \n");
	Print(b2);
	//printf("Dual int (b3) \n");
	//Print(b3);
}

__host__ __device__ void TryAddition()
{
#if defined (__CUDA_ARCH__)
	// Device code here
	printf(" TryAddition(): Function is called on the device \n");
#else
	// Host code here
	printf(" TryAddition(): Function is called on the host \n");
#endif
	Dual<N, double> a1 = { 2, 1 };
	Dual<N, double> b1 = { 10, 0 };

	int		b2 = 2;
	float	b3 = 3.0f;
	double  b4 = 5.0;

	printf("Dual + Dual (a1+b1)\n");
	Print(a1 + b1);
	printf("Dual + Dual (b1+a1)\n");
	Print(b1 + a1);

	printf("Dual + int (a1+b2) \n");
	Print(a1 + b2);
	printf("int + Dual (b2+a1) \n");
	Print(b2 + a1);

	printf("Dual + float (a1+b3) \n");
	Print(a1 + b3);
	printf("float + Dual (b3+a1) \n");
	Print(b3 + a1);

	printf("Dual + double (a1+b4) \n");
	Print(a1 + b4);
	printf("double + Dual (a1+b4) \n");
	Print(b4 + a1);


	printf("Mixed: b2 + a1 + b3 + b4 + b1 + b3\n");
	Print(b2 + a1 + b3 + b4 + b1);

	printf("Mixed: b4 + b1 + b2 + a1 + b3 + b2\n");
	Print(b4 + b1 + b2 + a1 + b3);
}

__host__ __device__ void TrySubtraction()
{
#if defined (__CUDA_ARCH__)
	// Device code here
	printf(" TrySubtraction(): Function is called on the device \n");
#else
	// Host code here
	printf(" TrySubtraction(): Function is called on the host \n");
#endif

	Dual<N, double> a1 = { 2, 2 };
	Dual<N, double> b1 = { 10, 1 };
	Dual<N, double> c1 = { 6, 0 };

	int		b2 = 3;
	float	b3 = 3.0f;
	double  b4 = 5.0;

	printf("Dual - Dual (a1-b1)\n");
	Print(a1 - b1);
	printf("Dual - Dual (b1-a1)\n");
	Print(b1 - a1);

	printf("Dual - Dual - Dual (a1-b1-c1)\n");
	Print(a1 - b1 - c1);
	printf("Dual - Dual - Dual (b1-c1-a1)\n");
	Print(b1 - c1 - a1);

	printf("Dual - int (a1-b2) \n");
	Print(a1 - b2);
	printf("int - Dual (b2-a1) \n");
	Print(b2 - a1);

	printf("Dual - float (a1-b3) \n");
	Print(a1 - b3);
	printf("float - Dual (b3-a1) \n");
	Print(b3 - a1);

	printf("Dual - double (a1-b4) \n");
	Print(a1 - b4);
	printf("double - Dual (a1-b4) \n");
	Print(b4 - a1);

	printf("Mixed: b2 + a1 + b3 + b4 + b1 + b3\n");
	Print(b2 - a1 - b3 - b4 - b1);

	printf("Mixed: b4 - b1 - b2 - a1 - b3 - b2\n");
	Print(b4 - b1 - b2 - a1 - b3);
}

__host__ __device__ void TryMultiplication()
{
#if defined (__CUDA_ARCH__)
	// Device code here
	printf(" TryMultiplication(): Function is called on the device \n");
#else
	// Host code here
	printf(" TryMultiplication(): Function is called on the host \n");
#endif

	Dual<N, double> a1 = { 5, 0 };
	Dual<N, double> b1 = { 5, 1 };
	Dual<N, double> c1 = { 2, 2 };

	printf("Dual*Dual: a1*b1\n");
	Print(a1*b1);
	printf("Dual*Dual: b1*a1\n");
	Print(b1*a1);

	printf("b1*(a1+c1)\n");
	Print(b1*(a1+c1));

	printf("(b1*a1)+c1)\n");
	Print((b1*a1)+c1);

	double	d1 = 5.6;
	float	d2 = 6.1f;
	int		d3 = 4;

	printf("Dual*double: a1 * d1 \n");
	Print(a1*d1);
	printf("Dual*double: d1 * a1 \n");
	Print(d1*a1);

	printf("Dual*float: a1 * d2 \n");
	Print(a1*d2);
	printf("Dual*float: d2 * a1 \n");
	Print(d2*a1);

	printf("Dual*int: a1 * d3 \n");
	Print(a1*d3);
	printf("Dual*int: d3 * a1 \n");
	Print(d3*a1);
}

__host__ __device__ void TryDivision()
{
#if defined (__CUDA_ARCH__)
	// Device code here
	printf(" TryDivision(): Function is called on the device \n");
#else
	// Host code here
	printf(" TryDivision(): Function is called on the host \n");
#endif

	Dual<N, double> a1 = { 5, 0 };
	Dual<N, double> b1 = { 2.5, 1 };
	Dual<N, double> c1 = { 2, 2 };

	printf("Dual/Dual: a1/b1\n");
	Print(a1/b1);
	printf("Dual/Dual: b1/a1\n");
	Print(b1/a1);

	printf("b1/(a1/c1)\n");
	Print(b1/(a1/c1));

	printf("(b1/a1)/c1)\n");
	Print((b1/a1)/c1);

	double	d1 = 5.6;
	float	d2 = 6.1f;
	int		d3 = 4;

	printf("Dual/double: a1 / d1 \n");
	Print(a1/d1);
	printf("Dual/double: d1 / a1 \n");
	Print(d1/a1);

	printf("Dual/float: a1 / d2 \n");
	Print(a1/d2);
	printf("Dual/float: d2 / a1 \n");
	Print(d2/a1);

	printf("Dual/int: a1 / d3 \n");
	Print(a1/d3);
	printf("Dual/int: d3 / a1 \n");
	Print(d3/a1);
}

__global__ void TryDualNumbers()
{
	//TryVariableInitialization();
	TryAddition();
	TrySubtraction();
	TryMultiplication();
	TryDivision();
}

int main()
{
	//TryVariableInitialization();
	TryAddition();
	TrySubtraction();
	TryMultiplication();
	TryDivision();

	TryDualNumbers KERNEL_ARGS2(1, 1)();

	return 0;
}