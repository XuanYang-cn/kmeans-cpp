#include <cstring>
#include <iostream>
#include <random>
#include "kmeans.h"

Clustering::Clustering(int64_t d, int64_t k, int64_t niter)
    : d(d),
      k(k),
      niter(niter) {
    
    centroids = new double *[k];
    for (int i = 0; i < k; i++) {
        centroids[i] = new double[d];
        memset( centroids[i], 0, sizeof(double) * d );
    }
}


void
Clustering::train( int64_t n, const double * x ) {
    // checking
    if (k > n) {
        std::cerr << "Number of centroids is larger than number of vectors!" << std::endl;
        exit(0);
    }

    // choose K distinct values for the centers of the clusters
    srand((unsigned) time(0));
    double *samples = new double[d];

    for( int i = 0; i < k; ++i ) {
        int64_t select = n/k * i + ( n/k - 1 ) * rand() / RAND_MAX;
        std::cout << "select: " << select << std::endl;
        for( int j = 0; j < d; ++j  ){
            samples[j] = x[select*d+j];
        }
        
        memcpy( centroids[i], samples, sizeof(double) * d);
    }
    
    for (int i = 0; i < k; i ++) {
    std::cout << "centroids:  ";
        for ( int j = 0; j < d; j++ ) {
            std::cout << centroids[i][j] <<" ";
        }
    std::cout << std::endl;
    }
    delete[] samples;


    // Recursion



}
