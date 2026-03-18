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

    static int computeSize(const std::vector<int>& layout);

    int computeLinearIndex(const std::vector<int>& index) const;
    std::vector<int> computeMultiIndex(int linearIndex) const;
    void ensureSameLayout(const Tensor& other, const char* op) const;

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

    Tensor operator+() const;
    Tensor operator-() const;

    Tensor operator+(double scalar) const;
    Tensor operator-(double scalar) const;
    Tensor operator*(double scalar) const;
    Tensor operator/(double scalar) const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor& operator+=(double scalar);
    Tensor& operator-=(double scalar);
    Tensor& operator*=(double scalar);
    Tensor& operator/=(double scalar);

    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    void fill(double value);

    Tensor reshape(const std::vector<int>& newLayout) const;
    Tensor transpose(int axis1, int axis2) const;
    Tensor flatten() const;
    Tensor clone() const;
    Tensor multiply(const Tensor& other) const;

    double sum() const;
    double mean() const;
    double min() const;
    double max() const;

    std::vector<double> toVector() const;

    int size() const;
    int rank() const;

    std::vector<int> getStrides() const;
    std::vector<int> getLayout() const;

    static Tensor zeros(const std::vector<int>& layout);
    static Tensor ones(const std::vector<int>& layout);
    static Tensor full(const std::vector<int>& layout, double value);
    static Tensor arange(double start, double end, double step = 1.0);

    friend Tensor operator+(double scalar, const Tensor& tensor);
    friend Tensor operator-(double scalar, const Tensor& tensor);
    friend Tensor operator*(double scalar, const Tensor& tensor);
    friend Tensor operator/(double scalar, const Tensor& tensor);

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
};

}
