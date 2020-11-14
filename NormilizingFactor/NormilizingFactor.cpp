#include <iostream>
#include <math.h>
#include <chrono> 
#include <stdlib.h>
#include <iomanip>
#include <omp.h>

int factorial(int i)
{
    if (i == 0) return 1;
    else return i * factorial(i - 1);
}

void output(std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end, double res) {
    double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    time_taken *= 1e-9;
    std::cout << "result: " << res << std::endl << " time : " << time_taken << std::setprecision(12) << " sec" << std::endl;
}

struct SMO {
public:
    int r;
    double e;
    double mu;

    SMO(){
        r = 1;
        e = 0.5;
        mu = 1;
    }

    SMO(int r, double e, double mu) {
        this->r = r;
        this->e = e;
        this->mu = mu;
    }

    double getP(int k) {
        double firstPart = pow(e / mu, k);

        double secondPart = 0;

        if (k <= r) {
            secondPart = 1.0 / factorial(k);
        }
        else {
            secondPart = 1.0 / (factorial(r) * pow(r, k - r));
        }

        return firstPart * secondPart;
    }
};

double find_normalizing_factor(SMO* smo, int n) {
    double almostRes = 0;

    for (int i = 0; i <= n; i++) {
        almostRes += smo[0].getP(i) * smo[1].getP(n - i);
    }

    return 1.0 / almostRes;
}

double find_normalizing_factor_parallel(SMO* smo, int n) {
    double almostRes = 0;
    int i;
#pragma omp parallel for private(i) reduction(+:almostRes) shared(smo, n)
    for ( i = 0; i <= n; i++) {
        almostRes += smo[0].getP(i) * smo[1].getP(n - i);
    }

    return 1.0 / almostRes;
}


int main()
{
    SMO* smo = new SMO[2] { SMO(1, 1, 1.000000000000123), SMO(1, 1, 0.999999952342) };

    int n = 2e8;

    auto start = std::chrono::high_resolution_clock::now();
    double res1 = find_normalizing_factor_parallel(smo, n);
    auto end = std::chrono::high_resolution_clock::now();
    output(start, end, res1);

    start = std::chrono::high_resolution_clock::now();
    double res2 = find_normalizing_factor(smo, n);
    end = std::chrono::high_resolution_clock::now();
    output(start, end, res2); 
}