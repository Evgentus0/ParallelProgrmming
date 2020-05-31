#include <iostream>
#include <math.h>

int factorial(int i)
{
    if (i == 0) return 1;
    else return i * factorial(i - 1);
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

    SMO(double r, double e, double mu) {
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


int main()
{
    SMO* smo = new SMO[2] { SMO(1, 1, 1.000000000123), SMO(1, 1, 0.999952342) };

    int n = 10000000;

    double res = find_normalizing_factor(smo, n);

    std::cout << res;
}