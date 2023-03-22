#ifndef ETL_h
#define ETL_h

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <vector>

class ETL
{
    std::string dataset;
    std::string delimiter;
    bool header;

public:

    ETL(std::string data, std::string separator, bool head) : dataset(data), delimiter(separator), header(head)
    {}

    std::vector<std::vector<std::string>> readCSV();

    Eigen::MatrixXd CSVtoEigen(std::vector<std::vector<std::string>> dataset, int rows, int cols);
};

#endif