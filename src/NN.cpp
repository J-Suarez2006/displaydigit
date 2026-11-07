#include "../include/NN.hpp"

Vec<10> DigitNN::forward(const Vec<784>& input)
{
    input_layer = input;

    pre_activation_1 = w1 * input + b1;

    hidden_layer = ReLU(pre_activation_1);
    output_layer = softmax(w2 * hidden_layer + b2);

    return output_layer;
}

Vec<784> getInputFromImage(digit_pixels d)
{
    Vec<784> input;

    int row_num {0};

    for (const auto& row : d)
    {
        int pixel_num {0};
        for (const auto& p : row)
        {
            input(row_num * 28 + pixel_num, 0) = (float)p / 255.0f;

            ++pixel_num;
        }
        ++row_num;
    }

    return input;
}


int getOutput(const Vec<10>& v)
{
    float max {v(0, 0)};
    int index {0};
    for (int i = 0; i < 10; ++i)
    {
        if (v(i, 0) > max)
        {
            index = i;
            max = v(i, 0);
        }
    }

    return index;
}


void loadNN(const std::string filename, DigitNN& NN)
{
    std::ifstream input(filename, std::ios::in | std::ios::binary);

    if (!input.is_open())
    {
        throw std::runtime_error("Error: Could not open output file!");
    }

    loadMat(input, NN.w1);
    loadMat(input, NN.b1);
    loadMat(input, NN.w2);
    loadMat(input, NN.b2);
}
