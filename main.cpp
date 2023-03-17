#include "ETL/ETL.h"

#include "iostream"
#include "string"
#include "Eigen/Dense"
#include <boost/algorithm/string.hpp>
#include "vector"

int main(int argc, char *argv[]){
    ETL etl(argv[1], argv[2], argv[3]);

    std::vector<std::vector<std::string>> dataset = etl.readCSV();

    return EXIT_SUCCESS;
}