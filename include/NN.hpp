#ifndef NN_HPP
#define NN_HPP

#include "math.hpp"
#include "parsemnist.hpp"

#include <fstream>
#include <random>

template <int n>
Vec<n> ReLU(const Vec<n>& v)
{
    Vec<n> res;
    
    for (int i = 0; i < n; ++i)
    {
        res(i, 0) = std::max(v(i, 0), 0.0f);
    }

    return res;
}

template <int n>
Vec<n> d_ReLU(const Vec<n>& v)
{
    Vec<n> res;
    
    for (int i = 0; i < n; ++i)
        res(i, 0) = (v(i, 0) > 0 ? 1 : 0); 

    return res;
}

template <int n>
Vec<n> softmax(const Vec<n>& v)
{
    float max_val = v(0, 0);

    for (int i = 1; i < n; ++i)
    {
        max_val = std::max(max_val, v(i, 0));
    }

    float sum {0.0f};
    
    for (int i = 0; i < n; ++i)
    {
        sum += std::exp(v(i, 0) - max_val);
    }

    Vec<n> res;

    for (int i = 0; i < n; ++i)
    {
        res(i, 0) = std::exp(v(i, 0) - max_val) / sum;
    }

    return res;
}

#pragma pack(push, 1)
class DigitNN
{
    public:
    Vec<784> input_layer;
    Vec<256> pre_activation_1;
    Vec<256> hidden_layer;
    Vec<10> output_layer;

    Mat<256, 784> w1;
    Vec<256> b1;

    Mat<10, 256> w2;
    Vec<10> b2;

    Vec<10> forward(const Vec<784>& input);
};
#pragma pack(pop)

Vec<784> getInputFromImage(digit_pixels d);

int getOutput(const Vec<10>& v);

template <int r, int c>
void loadMat(std::ifstream& input, Mat<r, c>& m)
{
    input.read(reinterpret_cast<char*>(&m(0, 0)), r * c * sizeof(float));
}

void loadNN(const std::string filename, DigitNN& NN);

#endif