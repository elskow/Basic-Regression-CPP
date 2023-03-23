#include "LinearRegression.h"

#include <iostream>
#include "eigen3/Eigen/Dense"
#include "cmath"
#include "vector"


float LinearRegression::OLS_Cost(Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::MatrixXd theta){
    Eigen::MatrixXd inner = pow(((x*theta)-y).array(),2);

    return (inner.sum()/(2*x.rows()));
}

std::tuple<Eigen::VectorXd, std::vector<float>> LinearRegression::GradientDecent(Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::VectorXd theta, float alpha, int iters){
    Eigen::MatrixXd temp = theta;

    int parameters = theta.rows();  

    std::vector<float> cost;
    cost.push_back(OLS_Cost(x,y,theta));

    for (int i = 0; i < iters; i++){
        Eigen::MatrixXd error = x*theta - y;
        for (int j = 0; j < parameters; j++){
            Eigen::MatrixXd X_i = x.col(j);
            Eigen::MatrixXd term =  error.cwiseProduct(X_i);
            temp(j,0) = theta(j,0) - alpha*(term.sum()/x.rows());
        }
        theta = temp;
        cost.push_back(OLS_Cost(x,y,theta));
    }

    return std::make_tuple(theta, cost);
}

float LinearRegression::RSquared(Eigen::MatrixXd y, Eigen::MatrixXd y_hat){
    auto num = pow((y-y_hat).array(),2).sum();
    auto den = pow(y.array() - y.mean(),2).sum();

    return 1 - num/den;
}