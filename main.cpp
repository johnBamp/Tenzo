#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "include/tensor.hpp"

namespace {

using tenzo::Tensor;

Tensor addBiasRows(const Tensor& matrix, const Tensor& bias) {
    const std::vector<int> matrixShape = matrix.getLayout();
    const std::vector<int> biasShape = bias.getLayout();

    if (matrixShape.size() != 2 || biasShape != std::vector<int>({1, matrixShape[1]})) {
        throw std::runtime_error("Bias shape must be [1, columns]");
    }

    Tensor result(matrixShape);
    for (int row = 0; row < matrixShape[0]; ++row) {
        for (int col = 0; col < matrixShape[1]; ++col) {
            result({row, col}) = matrix({row, col}) + bias({0, col});
        }
    }
    return result;
}

Tensor applySigmoid(const Tensor& tensor) {
    Tensor result(tensor.getLayout());
    const std::vector<int> shape = tensor.getLayout();

    if (shape.size() != 2) {
        throw std::runtime_error("Sigmoid helper expects a rank-2 tensor");
    }

    for (int row = 0; row < shape[0]; ++row) {
        for (int col = 0; col < shape[1]; ++col) {
            const double value = tensor({row, col});
            result({row, col}) = 1.0 / (1.0 + std::exp(-value));
        }
    }

    return result;
}

Tensor sigmoidDerivativeFromActivation(const Tensor& activation) {
    return activation.multiply(Tensor::ones(activation.getLayout()) - activation);
}

Tensor sumRows(const Tensor& tensor) {
    const std::vector<int> shape = tensor.getLayout();
    if (shape.size() != 2) {
        throw std::runtime_error("sumRows expects a rank-2 tensor");
    }

    Tensor result({1, shape[1]});
    result.fill(0.0);

    for (int row = 0; row < shape[0]; ++row) {
        for (int col = 0; col < shape[1]; ++col) {
            result({0, col}) += tensor({row, col});
        }
    }

    return result;
}

void printPredictions(const Tensor& inputs, const Tensor& predictions) {
    std::cout << "Predictions\n";
    for (int row = 0; row < inputs.getLayout()[0]; ++row) {
        const double x1 = inputs({row, 0});
        const double x2 = inputs({row, 1});
        const double probability = predictions({row, 0});
        const int label = probability >= 0.5 ? 1 : 0;
        std::cout << static_cast<int>(x1) << " xor " << static_cast<int>(x2)
                  << " -> " << std::fixed << std::setprecision(4) << probability
                  << " (" << label << ")\n";
    }
}

} // namespace

int main() {
    Tensor inputs({4, 2});
    inputs = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    };

    Tensor targets({4, 1});
    targets = {
        0.0,
        1.0,
        1.0,
        0.0
    };

    Tensor w1({2, 4});
    w1 = {
        3.2, -3.1, 2.7, -2.8,
        3.3, -3.0, -2.6, 2.9
    };

    Tensor b1({1, 4});
    b1 = {
        -1.5, 4.5, -1.4, 1.2
    };

    Tensor w2({4, 1});
    w2 = {
        4.8,
        -6.1,
        4.2,
        4.4
    };

    Tensor b2({1, 1});
    b2 = { -1.9 };

    const double learningRate = 0.8;
    const double scale = 1.0 / static_cast<double>(inputs.getLayout()[0]);
    const int epochs = 6000;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        const Tensor z1 = addBiasRows(inputs * w1, b1);
        const Tensor a1 = applySigmoid(z1);
        const Tensor z2 = addBiasRows(a1 * w2, b2);
        const Tensor predictions = applySigmoid(z2);

        const Tensor outputError = predictions - targets;
        const Tensor dOutput = outputError.multiply(sigmoidDerivativeFromActivation(predictions));

        const Tensor hiddenError = dOutput * w2.transpose(0, 1);
        const Tensor dHidden = hiddenError.multiply(sigmoidDerivativeFromActivation(a1));

        const Tensor gradW2 = a1.transpose(0, 1) * dOutput * scale;
        const Tensor gradB2 = sumRows(dOutput) * scale;
        const Tensor gradW1 = inputs.transpose(0, 1) * dHidden * scale;
        const Tensor gradB1 = sumRows(dHidden) * scale;

        w2 -= gradW2 * learningRate;
        b2 -= gradB2 * learningRate;
        w1 -= gradW1 * learningRate;
        b1 -= gradB1 * learningRate;

        if (epoch % 1000 == 0 || epoch == epochs - 1) {
            const Tensor diff = predictions - targets;
            const Tensor squared = diff.multiply(diff);
            std::cout << "epoch " << epoch
                      << " loss=" << std::fixed << std::setprecision(6)
                      << squared.mean() << "\n";
        }
    }

    const Tensor hidden = applySigmoid(addBiasRows(inputs * w1, b1));
    const Tensor output = applySigmoid(addBiasRows(hidden * w2, b2));

    std::cout << "\n";
    printPredictions(inputs, output);

    return 0;
}
