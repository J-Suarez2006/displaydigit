#ifndef NN_HPP
#define NN_HPP

#include "../include/math.hpp"
#include "../include/parsemnist.hpp"

#include <fstream>
#include <random>

template <int r, int c>
void heInit(Mat<r, c>& m);

template <int r, int c>
void xavierInit(Mat<r, c>& m);

#pragma pack(push, 1)
struct Gradient 
{
    Mat<256, 784> d_w1;
    Vec<256> d_b1;

    Mat<10, 256> d_w2;
    Vec<10> d_b2;

    Vec<10> loss;
};
#pragma pack(pop)

inline Gradient operator+(const Gradient& g1, const Gradient& g2)
{
    return Gradient{
        g1.d_w1 + g2.d_w1,
        g1.d_b1 + g2.d_b1,
        g1.d_w2 + g2.d_w2,
        g1.d_b2 + g2.d_b2,
        g1.loss + g2.loss
    };
}

inline Gradient operator*(float scalar, const Gradient& g)
{
    return Gradient{
        scalar * g.d_w1,
        scalar * g.d_b1,
        scalar * g.d_w2,
        scalar * g.d_b2,
        scalar * g.loss
    };
}

template <int n>
Vec<n> ReLU(const Vec<n>& v);

template <int n>
Vec<n> d_ReLU(const Vec<n>& v);

template <int n>
Vec<n> softmax(const Vec<n>& v);

#pragma pack(push, 1)
class DigitNN
{
    public:
    DigitNN()
    {
        heInit(w1);
        xavierInit(w2);
    }

    Vec<784> input_layer;
    Vec<256> pre_activation_1;
    Vec<256> hidden_layer;
    Vec<10> output_layer;

    Mat<256, 784> w1;
    Vec<256> b1;

    Mat<10, 256> w2;
    Vec<10> b2;
    
    void descend(const Gradient& g, float step_size);

    Vec<10> forward(const Vec<784>& input);

    Gradient backward(const Vec<10>& actual);
};
#pragma pack(pop)

Vec<784> getInputFromImage(digit_pixels d);

Vec<10> getLabelVactor(uint8_t value);

int getOutput(const Vec<10>& v);

template <int r, int c>
void saveMat(std::ofstream& output, const Mat<r, c>& m);

void saveNN(const std::string filename, const DigitNN& NN);

template <int r, int c>
void loadMat(std::ifstream& input, Mat<r, c>& m);

void loadNN(const std::string filename, DigitNN& NN);

#endif