#include "../include/tensor.hpp"
#include <algorithm>

using namespace tenzo;

Tensor::Tensor(const std::vector<int>& layout) : layout(layout) {
    data.resize(computeSize(layout), 0.0);

    strides.resize(layout.size());
    int stride = 1;
    for (int i = static_cast<int>(layout.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= layout[i];
    }
}

//-------------Helpers-------------
int Tensor::computeSize(const std::vector<int>& layout) {
    if (layout.empty()) {
        throw std::runtime_error("Layout cannot be empty");
    }

    int totalSize = 1;
    for (int dim : layout) {
        if (dim <= 0) {
            throw std::runtime_error("All dimensions must be positive");
        }
        totalSize *= dim;
    }

    return totalSize;
}

int Tensor::computeLinearIndex(const std::vector<int>& index) const {
    if (index.size() != layout.size()) {
        throw std::runtime_error("Index does not match dimensions");
    }

    int linear = 0;
    for (size_t i = 0; i < layout.size(); ++i) {
        if (index[i] < 0 || index[i] >= layout[i]) {
            throw std::runtime_error("Index out of bounds");
        }
        linear += index[i] * strides[i];
    }

    return linear;
}

std::vector<int> Tensor::computeMultiIndex(int linearIndex) const {
    if (linearIndex < 0 || linearIndex >= size()) {
        throw std::runtime_error("Linear index out of bounds");
    }

    std::vector<int> index(layout.size());
    for (size_t i = 0; i < layout.size(); ++i) {
        index[i] = linearIndex / strides[i];
        linearIndex %= strides[i];
    }

    return index;
}

void Tensor::ensureSameLayout(const Tensor& other, const char* op) const {
    if (layout != other.layout) {
        throw std::runtime_error(std::string("Tensor layouts do not match for ") + op);
    }
}

void Tensor::printRecursive(
    std::ostream& os,
    std::vector<int>& index,
    int dim
) const {
    if (dim == static_cast<int>(layout.size())) {
        os << (*this)(index);
        return;
    }

    os << "[";

    for (int i = 0; i < layout[dim]; ++i) {
        index.push_back(i);
        printRecursive(os, index, dim + 1);
        index.pop_back();

        if (i != layout[dim] - 1) {
            if (dim == static_cast<int>(layout.size()) - 1) {
                os << ", ";
            } else {
                os << ",\n";
                for (int s = 0; s < dim + 1; ++s) {
                    os << " ";
                }
            }
        }
    }

    os << "]";
}

void Tensor::matmulRecursive(
    const Tensor& other,
    Tensor& result,
    std::vector<int>& batchIndex,
    int dim
) const {
    int batchDims = static_cast<int>(layout.size()) - 2;

    if (dim == batchDims) {
        int m = layout[layout.size() - 2];
        int n = layout[layout.size() - 1];
        int p = other.layout[other.layout.size() - 1];

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                double sum = 0.0;

                for (int k = 0; k < n; ++k) {
                    std::vector<int> leftIndex = batchIndex;
                    std::vector<int> rightIndex = batchIndex;
                    std::vector<int> resultIndex = batchIndex;

                    leftIndex.push_back(i);
                    leftIndex.push_back(k);

                    rightIndex.push_back(k);
                    rightIndex.push_back(j);

                    resultIndex.push_back(i);
                    resultIndex.push_back(j);

                    sum += (*this)(leftIndex) * other(rightIndex);
                    result(resultIndex) = sum;
                }
            }
        }
        return;
    }

    for (int b = 0; b < layout[dim]; ++b) {
        batchIndex.push_back(b);
        matmulRecursive(other, result, batchIndex, dim + 1);
        batchIndex.pop_back();
    }
}

//-------------Operators-------------
std::ostream& tenzo::operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(shape=[";
    for (size_t i = 0; i < tensor.layout.size(); ++i) {
        os << tensor.layout[i];
        if (i != tensor.layout.size() - 1) {
            os << ", ";
        }
    }
    os << "], data=\n";

    std::vector<int> index;
    tensor.printRecursive(os, index, 0);

    os << ")";
    return os;
}

double& Tensor::operator()(const std::vector<int>& index) {
    return data[computeLinearIndex(index)];
}

const double& Tensor::operator()(const std::vector<int>& index) const {
    return data[computeLinearIndex(index)];
}

// = Operators
Tensor& Tensor::operator=(const std::vector<double>& newData) {
    if (data.size() != newData.size()) {
        throw std::runtime_error("Dimensions do not match");
    }
    data = newData;
    return *this;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }

    data = other.data;
    layout = other.layout;
    strides = other.strides;

    return *this;
}

Tensor Tensor::operator+() const {
    return clone();
}

Tensor Tensor::operator-() const {
    Tensor result(layout);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = -data[i];
    }
    return result;
}

Tensor Tensor::operator+(double scalar) const {
    Tensor result(layout);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + scalar;
    }
    return result;
}

Tensor Tensor::operator-(double scalar) const {
    Tensor result(layout);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] - scalar;
    }
    return result;
}

// * Operators
Tensor Tensor::operator*(double scalar) const {
    Tensor result(layout);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Tensor Tensor::operator/(double scalar) const {
    if (scalar == 0.0) {
        throw std::runtime_error("Division by zero");
    }

    Tensor result(layout);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] / scalar;
    }
    return result;
}

Tensor Tensor::operator+(const Tensor& other) const {
    ensureSameLayout(other, "addition");

    Tensor result(layout);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    ensureSameLayout(other, "subtraction");

    Tensor result(layout);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (layout.size() < 2 || other.layout.size() < 2) {
        throw std::runtime_error("Matrix multiplication requires tensors with rank >= 2");
    }

    if (layout.size() != other.layout.size()) {
        throw std::runtime_error("Tensor ranks must match for batched matrix multiplication");
    }

    int rank = static_cast<int>(layout.size());

    // Check batch dimensions
    for (int i = 0; i < rank - 2; ++i) {
        if (layout[i] != other.layout[i]) {
            throw std::runtime_error("Batch dimensions do not match");
        }
    }

    int m = layout[rank - 2];
    int n = layout[rank - 1];
    int otherN = other.layout[rank - 2];
    int p = other.layout[rank - 1];

    if (n != otherN) {
        throw std::runtime_error("Inner matrix dimensions do not match");
    }

    std::vector<int> resultLayout;
    for (int i = 0; i < rank - 2; ++i) {
        resultLayout.push_back(layout[i]);
    }
    resultLayout.push_back(m);
    resultLayout.push_back(p);

    Tensor result(resultLayout);

    std::vector<int> batchIndex;
    matmulRecursive(other, result, batchIndex, 0);

    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    ensureSameLayout(other, "division");

    Tensor result(layout);
    for (size_t i = 0; i < data.size(); ++i) {
        if (other.data[i] == 0.0) {
            throw std::runtime_error("Division by zero");
        }
        result.data[i] = data[i] / other.data[i];
    }
    return result;
}

Tensor& Tensor::operator+=(double scalar) {
    for (double& value : data) {
        value += scalar;
    }
    return *this;
}

Tensor& Tensor::operator-=(double scalar) {
    for (double& value : data) {
        value -= scalar;
    }
    return *this;
}

Tensor& Tensor::operator*=(double scalar) {
    for (double& value : data) {
        value *= scalar;
    }
    return *this;
}

Tensor& Tensor::operator/=(double scalar) {
    if (scalar == 0.0) {
        throw std::runtime_error("Division by zero");
    }

    for (double& value : data) {
        value /= scalar;
    }
    return *this;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    ensureSameLayout(other, "addition");

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    ensureSameLayout(other, "subtraction");

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= other.data[i];
    }
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    ensureSameLayout(other, "division");

    for (size_t i = 0; i < data.size(); ++i) {
        if (other.data[i] == 0.0) {
            throw std::runtime_error("Division by zero");
        }
        data[i] /= other.data[i];
    }
    return *this;
}

//-------------Misc-------------
void Tensor::fill(double value) {
    std::fill(data.begin(), data.end(), value);
}

Tensor Tensor::reshape(const std::vector<int>& newLayout) const {
    if (computeSize(newLayout) != size()) {
        throw std::runtime_error("Reshape size must match tensor size");
    }

    Tensor result(newLayout);
    result.data = data;
    return result;
}

Tensor Tensor::transpose(int axis1, int axis2) const {
    int tensorRank = rank();
    if (axis1 < 0 || axis1 >= tensorRank || axis2 < 0 || axis2 >= tensorRank) {
        throw std::runtime_error("Transpose axes out of bounds");
    }

    std::vector<int> resultLayout = layout;
    std::swap(resultLayout[axis1], resultLayout[axis2]);

    Tensor result(resultLayout);

    for (int linearIndex = 0; linearIndex < size(); ++linearIndex) {
        std::vector<int> index = computeMultiIndex(linearIndex);
        std::swap(index[axis1], index[axis2]);
        result(index) = data[linearIndex];
    }

    return result;
}

Tensor Tensor::flatten() const {
    return reshape({size()});
}

Tensor Tensor::clone() const {
    Tensor result(layout);
    result.data = data;
    return result;
}

Tensor Tensor::multiply(const Tensor& other) const {
    ensureSameLayout(other, "elementwise multiplication");

    Tensor result(layout);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

double Tensor::sum() const {
    double result = 0.0;
    for (double value : data) {
        result += value;
    }
    return result;
}

double Tensor::mean() const {
    return sum() / static_cast<double>(size());
}

double Tensor::min() const {
    return *std::min_element(data.begin(), data.end());
}

double Tensor::max() const {
    return *std::max_element(data.begin(), data.end());
}

std::vector<double> Tensor::toVector() const {
    return data;
}

int Tensor::size() const {
    return static_cast<int>(data.size());
}

int Tensor::rank() const {
    return static_cast<int>(layout.size());
}

std::vector<int> Tensor::getStrides() const {
    return strides;
}

std::vector<int> Tensor::getLayout() const {
    return layout;
}

Tensor Tensor::zeros(const std::vector<int>& layout) {
    return Tensor(layout);
}

Tensor Tensor::ones(const std::vector<int>& layout) {
    return full(layout, 1.0);
}

Tensor Tensor::full(const std::vector<int>& layout, double value) {
    Tensor tensor(layout);
    tensor.fill(value);
    return tensor;
}

Tensor Tensor::arange(double start, double end, double step) {
    if (step == 0.0) {
        throw std::runtime_error("Step cannot be zero");
    }

    std::vector<double> values;

    if (step > 0.0) {
        for (double value = start; value < end; value += step) {
            values.push_back(value);
        }
    } else {
        for (double value = start; value > end; value += step) {
            values.push_back(value);
        }
    }

    if (values.empty()) {
        throw std::runtime_error("arange produced an empty tensor");
    }

    Tensor tensor({static_cast<int>(values.size())});
    tensor.data = values;
    return tensor;
}

Tensor tenzo::operator+(double scalar, const Tensor& tensor) {
    return tensor + scalar;
}

Tensor tenzo::operator-(double scalar, const Tensor& tensor) {
    Tensor result(tensor.layout);
    for (size_t i = 0; i < tensor.data.size(); ++i) {
        result.data[i] = scalar - tensor.data[i];
    }
    return result;
}

Tensor tenzo::operator*(double scalar, const Tensor& tensor) {
    return tensor * scalar;
}

Tensor tenzo::operator/(double scalar, const Tensor& tensor) {
    Tensor result(tensor.layout);
    for (size_t i = 0; i < tensor.data.size(); ++i) {
        if (tensor.data[i] == 0.0) {
            throw std::runtime_error("Division by zero");
        }
        result.data[i] = scalar / tensor.data[i];
    }
    return result;
}
