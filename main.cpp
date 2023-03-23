#include "ETL/ETL.h"
#include "LinearRegression/LinearRegression.h"

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

    Eigen::VectorXd vec_train = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vec_test = Eigen::VectorXd::Ones(X_test.rows());

    X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
    X_train.col(X_train.cols()-1) = vec_train;

    X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
    X_test.col(X_test.cols()-1) = vec_test;

    LinearRegression lr;

    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());  
    float alpha = 0.01;
    int iterations = 1000;

    Eigen::VectorXd thetaOut;
    std::vector<float> cost;

    auto mu_data = etl.Mean(dataMat);
    auto mu_z = mu_data(0,11);

    auto scaled_data = dataMat.rowwise() - dataMat.colwise().mean();

    auto sigma_data = etl.Std(scaled_data);
    auto sigma_z = sigma_data(0,11);

    Eigen::MatrixXd y_train_hat = (X_train*thetaOut*sigma_z).array() + mu_z;
    Eigen::MatrixXd y = dataMat.col(11).topRows(1279);

    float R_Squared = lr.RSquared(y,y_train_hat);
    std::cout << "R Squared :" << R_Squared << std::endl;
    
    return EXIT_SUCCESS;
}