# include <stddef.h>
# include <vector>
# include <cstdint> // int64_t

struct
Clustering {
    size_t d; // dimension of the vectors
    size_t k; // number of centroids
    int64_t niter; // number of iteration times


    double** centroids; // (k * d) centroids

    Clustering(int64_t d, int64_t k, int64_t niter);


    ~Clustering() {
        if ( centroids != nullptr ) {
            for (int i = 0; i < k; i++){
                if (centroids[i] != nullptr)
                    delete[] centroids[i];
            }
        }
        delete[] centroids;
    }
    /**
     *
     * @param x     training vectors, size n * d
     * @param n     number of the vectors
     * */
    virtual void
    train( int64_t n, const double * x);
};
