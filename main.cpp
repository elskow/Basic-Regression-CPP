#include "ETL/ETL.h"

#include "iostream"
#include "string"
#include "eigen3/Eigen/Dense"
#include <boost/algorithm/string.hpp>
#include "vector"

int main(int argc, char *argv[]){
    ETL etl(argv[1], argv[2], argv[3]);

    std::vector<std::vector<std::string>> dataset = etl.readCSV();

    int rows = dataset.size();
    int cols = dataset[0].size();

    Eigen::MatrixXd dataMat = etl.CSVtoEigen(dataset, rows, cols);
    Eigen::MatrixXd normData = etl.Normalize(dataMat);
    
    Eigen::MatrixXd X_train, y_train, X_test, y_test;

    std::tie(X_train, y_train, X_test, y_test) = etl.TrainTestSplit(dataMat, 0.8);

    std::cout << "X_train: " << X_train.size() << std::endl;
    std::cout << "y_train: " << y_train.size() << std::endl;

    std::cout << "X_test: " << X_test.size() << std::endl;
    std::cout << "y_test: " << y_test.size() << std::endl;

    return EXIT_SUCCESS;
}