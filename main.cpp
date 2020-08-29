#include <iostream>
#include "kmeans.h"

int
main() {
    int dimention = 2;
    int k = 1;
    Clustering cluster( dimention, k );

    std::cout << cluster.d << ","
              << cluster.k << std::endl;
}
