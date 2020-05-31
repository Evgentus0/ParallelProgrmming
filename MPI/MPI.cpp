#include<iostream> 
#include<mpi.h>  
#include <chrono> 
#include <stdlib.h>
#include <iomanip>

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

    SMO() {
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


double parallel_normalizing_factor(int rank, int comm_size, int n, SMO* smo) {
    int left;
    int right;

    if (rank == 0) {
        double result = 0;
        int length;
        length = n / comm_size;

        for (int i = 1; i < comm_size; i++) {
            left = i * length;
            right = __min(n, left + length);
            MPI_Send(&left,
                1, MPI_DOUBLE,
                i, i,
                MPI_COMM_WORLD);
            MPI_Send(&right,
                1, MPI_DOUBLE,
                i, i,
                MPI_COMM_WORLD);
        }

        left = 0;
        right = length;
        for (int i = left; i <= right; i++)
        {
            result += smo[0].getP(i) * smo[1].getP(n - i);
        }

        double SUB_RESULT;

        for (int i = 1; i < comm_size; i++) {
            MPI_Recv(&SUB_RESULT, 1, MPI_DOUBLE,
                i, i,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);

            result += SUB_RESULT;
        }

        return 1/result;
    }
    else {
        double sub_result = 0;
        MPI_Recv(&left, 1, MPI_DOUBLE,
            0, rank,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
        MPI_Recv(&right, 1, MPI_DOUBLE,
            0, rank,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
        for (int i = left; i <= right; i++)
        {
            sub_result += smo[0].getP(i) * smo[1].getP(n - i);
        }
        MPI_Send(&sub_result,
            1, MPI_DOUBLE,
            0, rank,
            MPI_COMM_WORLD);
    }

    return 0;
}

int main() {   
    SMO* smo = new SMO[2]{ SMO(1, 1, 1.000000000000123), SMO(1, 1, 0.999999952342) };

    int n = 1e9;

	int rank, comm_size, t;   

	MPI_Init(NULL, NULL);   
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);   
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);   

    auto start = std::chrono::high_resolution_clock::now();
    double result = parallel_normalizing_factor(rank, comm_size, n, smo);
    auto end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        output(start, end, result);
    }

	MPI_Finalize();   
	return 0;  
}