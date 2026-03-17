#pragma once
#include <iostream>
#include <vector>
#include <stdexcept>

namespace tenzo {

class Tensor {
private:
    std::vector<double> data;
    std::vector<int> layout;
    std::vector<int> strides;

    int computeLinearIndex(const std::vector<int>& index) const;

    void matmulRecursive(
        const Tensor& other,
        Tensor& result,
        std::vector<int>& batchIndex,
        int dim
    ) const;

    void printRecursive(
        std::ostream& os,
        std::vector<int>& index,
        int dim
    ) const;

public:
    Tensor(const std::vector<int>& layout);

    double& operator()(const std::vector<int>& index);
    const double& operator()(const std::vector<int>& index) const;

    Tensor& operator=(const std::vector<double>& newData);
    Tensor& operator=(const Tensor& other);

    Tensor operator*(double scalar) const;
    Tensor operator*(const Tensor& other) const;

    int size() const;

    std::vector<int> getStrides() const;
    std::vector<int> getLayout() const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
};

}