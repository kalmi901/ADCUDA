#ifndef AD_H
#define AD_H

#include <cuda_runtime.h>

// Struct to represent Dual numbers
template <size_t NUMVAR, class Precision>
struct Dual
{
public:
	// store real and dual values
	Precision			 real;
	Precision dual[NUMVAR] {};

	// constructor
	__host__ __device__ Dual(Precision value, int i)
		: real(value)	{dual[i] = (Precision)1.0;	}

	__host__ __device__ Dual(Precision value)
		:real(value)	{}
};


//----------------- Math Operators ----------------------------

// ADD --------------------------------------------------------
// Dual + Dual
template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator + (const Dual<NUMVAR, Precision> &a, const Dual<NUMVAR, Precision> &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = a.real + b.real;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i] + b.dual[i];
	}
	return c;
}

// Dual + Number
template<size_t NUMVAR, class Precision, class OtherPrecision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator + (const Dual<NUMVAR, Precision> &a, const OtherPrecision &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = a.real + (Precision)b;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i];
	}
	return c;
}

// Number + Dual
template<size_t NUMVAR, class Precision, class OtherPrecision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator + (const OtherPrecision &a, const Dual<NUMVAR, Precision> &b)
{
	return b + a;
}

// Substract ----------------------------------------------------------
// Dual - Dual
template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator - (const Dual<NUMVAR, Precision> &a, const Dual<NUMVAR, Precision> &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = a.real - b.real;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i] - b.dual[i];
	}
	return c;
}

// Dual - Number
template<size_t NUMVAR, class Precision, class OtherPrecision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator - (const Dual<NUMVAR, Precision> &a, const OtherPrecision &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = a.real - (Precision)b;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i];
	}
	return c;
}

// Number - Dual
template<size_t NUMVAR, class Precision, class OtherPrecision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator - (const OtherPrecision &a, const Dual<NUMVAR, Precision> &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = (Precision)a - b.real;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = -b.dual[i];
	}
	return c;
}

// Multiply -----------------------------------------------------------
// Dual * Dual
template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator * (const Dual<NUMVAR, Precision> &a, const Dual<NUMVAR, Precision> &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = a.real * b.real;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.real * b.dual[i] + a.dual[i] * b.real;
	}
	return c;
}

// Dual * Number
template<size_t NUMVAR, class Precision, class OtherPrecision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator * (const Dual<NUMVAR, Precision> &a, const OtherPrecision &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = a.real * (Precision)b;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i] * (Precision)b;
	}
	return c;
}

// Number * Dual
template<size_t NUMVAR, class Precision, class OtherPrecision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator * (const OtherPrecision &a, const Dual<NUMVAR, Precision> &b)
{
	return b * a;
}

// Divide -----------------------------------------------
// Dual / Dual
template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator / (const Dual<NUMVAR, Precision> &a, const Dual<NUMVAR, Precision> &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = a.real / b.real;
	Precision rD = (Precision)1.0 / (b.real * b.real);
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = (a.dual[i] * b.real - a.real * b.dual[i]) * rD;
	}
	return c;
}

// Dual / Number
template<size_t NUMVAR, class Precision, class OtherPrecision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator / (const Dual<NUMVAR, Precision> &a, const OtherPrecision &b)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = a.real / (Precision)b;
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i] / (Precision)b;
	}
	return c;
}

// Number / Dual
template<size_t NUMVAR, class Precision, class OtherPrecision>
inline __host__ __device__ Dual<NUMVAR, Precision> operator / (const OtherPrecision &a, const Dual<NUMVAR, Precision> &b)
{
	//Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)a);
	//return c / b;
	return Dual<NUMVAR, Precision>((Precision)a) / b;
}

// ------------------ Functions -------------------------

// pow(), sqrt()
// log(), log10(), exp()
// sin(), cos(), tan(), ctan()


// Trigonometric functions

template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> sin(const Dual<NUMVAR, Precision> &a)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = sin(a.real);
	Precision Cos = cos(a.real);
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i] * Cos;
	}
	return c;
}

template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> cos(const Dual<NUMVAR, Precision> &a)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = cos(a.real);
	Precision Sin = sin(a.real);
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = -a.dual[i] * Sin;
	}
	return c;
}

template<size_t NUMVAR, class Precision>
inline __host__ __device__ Dual<NUMVAR, Precision> tan(const Dual<NUMVAR, Precision> &a)
{
	Dual<NUMVAR, Precision> c = Dual<NUMVAR, Precision>((Precision)0.0);
	c.real = tan(a.real);
	Precision Temp = (Precision)1.0/(cos(a.real) * cos(a.real));
	for (size_t i = 0; i < NUMVAR; i++)
	{
		c.dual[i] = a.dual[i] * Temp;
	}
	return c;
}

// asin(), acos(), atan(), actan()

#endif