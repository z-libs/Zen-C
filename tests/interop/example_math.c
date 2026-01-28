#include "example_math.h"

// Constants as doubles (enough for a decent approximation)
static const double PI      = 3.14159265358979323846264338327950288;
static const double TWO_PI  = 6.28318530717958647692528676655900576;
static const double HALF_PI = 1.57079632679489661923132169163975144;

// Simple int abs (no overflow handling for INT_MIN; fine for most tests)
int abs(int x) {
    return (x < 0) ? -x : x;
}

// Reduce x into [-PI, PI] without relying on libm (no fmod/floor needed).
static double wrap_pi(double x) {
    // Bring x into roughly [-TWO_PI, TWO_PI] by subtracting multiples of TWO_PI
    // using truncation toward zero (cast to long long).
    // This is "good enough" for typical game/math usage and simple tests.
    double k_real = x / TWO_PI;
    long long k = (long long)k_real; // trunc toward 0
    x = x - (double)k * TWO_PI;

    // Fix-up to ensure x is in [-PI, PI]
    if (x > PI)  x -= TWO_PI;
    if (x < -PI) x += TWO_PI;
    return x;
}

// Sine approximation:
// 1) range-reduce to [-PI, PI]
// 2) fold to [-PI/2, PI/2] using symmetries
// 3) evaluate odd polynomial (Taylor-ish) on that small range
double sin(double x) {
    // Ensure exact 0 stays exact 0 (nice for your test)
    if (x == 0.0) return 0.0;

    // Range reduce to [-PI, PI]
    x = wrap_pi(x);

    // Fold into [-PI/2, PI/2] for better accuracy
    // sin(x) =  sin(x)                    for x in [-PI/2, PI/2]
    // sin(x) =  sin(PI - x)               for x in ( PI/2, PI]
    // sin(x) = -sin(-PI - x)              for x in [-PI, -PI/2)
    int sign = 1;
    if (x > HALF_PI) {
        x = PI - x;
    } else if (x < -HALF_PI) {
        x = -PI - x;
        sign = -1;
    }

    // Polynomial approximation on [-PI/2, PI/2]
    // sin(x) ≈ x + x^3*c3 + x^5*c5 + x^7*c7 + x^9*c9 + x^11*c11
    // Coefficients from the Taylor series (good enough here).
    double x2 = x * x;

    const double c3  = -1.0 / 6.0;        // -0.16666666666666666
    const double c5  =  1.0 / 120.0;      //  0.008333333333333333
    const double c7  = -1.0 / 5040.0;     // -0.0001984126984126984
    const double c9  =  1.0 / 362880.0;   //  2.7557319223985893e-06
    const double c11 = -1.0 / 39916800.0; // -2.505210838544172e-08

    // Horner form for fewer multiplies:
    double poly = (((c11 * x2 + c9) * x2 + c7) * x2 + c5) * x2 + c3;
    double y = x + (x * x2) * poly;

    return (double)sign * y;
}
