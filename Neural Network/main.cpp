#include "include/math.hpp"
#include "include/parsemnist.hpp"
#include "include/NN.hpp"

#include <cmath>
#include <random>
#include <algorithm>
#include <string>
#include <numeric>

std::mt19937 gen(std::random_device{}());

template <int r, int c>
void heInit(Mat<r, c>& m)
{
    float stddev = std::sqrt(2.0f / c);
    std::normal_distribution<float> dist(0.0f, stddev);

    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            m(i, j) = dist(gen);
}

template <int r, int c>
void xavierInit(Mat<r, c>& m)
{
    float stddev = std::sqrt(1.0f / c);
    std::normal_distribution<float> dist(0.0f, stddev);

    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            m(i, j) = dist(gen);
}

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

    Vec<10> forward(const Vec<784>& input)
    {
        input_layer = input;

        pre_activation_1 = w1 * input + b1;

        hidden_layer = ReLU(pre_activation_1);
        output_layer = softmax(w2 * hidden_layer + b2);

        return output_layer;
    }

    Gradient backward(const Vec<10>& actual)
    {
        Gradient res;

        Vec<10> d2 = output_layer - actual;
        res.loss = d2;

        res.d_b2 = d2;
        res.d_w2 = d2 * transpose(hidden_layer);

        Vec<256> d1 = elementwiseMult(transpose(w2) * d2, d_ReLU(pre_activation_1));

        res.d_b1 = d1;
        res.d_w1 = d1 * transpose(input_layer);

        return res;
    }
};
#pragma pack(pop)

void DigitNN::descend(const Gradient& g, float step_size)
{
    w1 = w1 - step_size * g.d_w1;
    b1 = b1 - step_size * g.d_b1;
    w2 = w2 - step_size * g.d_w2; 
    b2 = b2 - step_size * g.d_b2; 
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

Vec<10> getLabelVactor(uint8_t value)
{
    if (value > 9) throw std::runtime_error("Something is fucked with the labels");

    Vec<10> res;
    res(value, 0) = 1.0f;

    return res;
}

template <int n>
float magnitude(const Vec<n>& v)
{
    float sum {0.0f};
    for (int i = 0; i < n; ++i)
    {
        sum += v(i, 0) * v(i, 0);
    }

    return std::sqrt(sum);

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

template <int r, int c>
void saveMat(std::ofstream& output, const Mat<r, c>& m)
{
    output.write(reinterpret_cast<const char*>(&m(0, 0)), r * c * sizeof(float));
}

void saveNN(const std::string filename, const DigitNN& NN)
{
    std::ofstream output(filename, std::ios::out | std::ios::binary);

    if (!output.is_open())
    {
        throw std::runtime_error("Error: Could not open output file!");
    }

    saveMat(output, NN.w1);
    saveMat(output, NN.b1);
    saveMat(output, NN.w2);
    saveMat(output, NN.b2);
}

template <int r, int c>
void loadMat(std::ifstream& input, Mat<r, c>& m)
{
    input.read(reinterpret_cast<char*>(&m(0, 0)), r * c * sizeof(float));
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

int main()
{
    auto images = parseMnistImages("mnist_data/train-images.idx3-ubyte");
    auto labels = parseMnistLabels("mnist_data/train-labels.idx1-ubyte");

    std::vector<int> indices(images.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
   
    constexpr int BATCH_SIZE = 100;
    constexpr int NUM_IMAGES = 60000;
    float STEP_SIZE = .01f;

    DigitNN NN;
    loadNN("digit_nn.nn", NN);
    
    for (int batch = 0; batch < NUM_IMAGES / BATCH_SIZE; ++batch)
    {
        std::cout << "Batch number" << batch << ", ";
        std::cout << (NUM_IMAGES / BATCH_SIZE) - batch << " batches remaining" << std::endl;
        Gradient batch_gradient {};
        float batch_loss {0.0f};

        for (int i = 0; i < BATCH_SIZE; ++i)
        {
            int index {indices[batch * BATCH_SIZE + i]};

            Vec<10> actual {getLabelVactor(labels[index])};
            Vec<784> img_input {getInputFromImage(images[index])};

            Vec<10> output = NN.forward(img_input);
            Gradient g = NN.backward(actual);

            batch_loss += -std::log(output(labels[index], 0) + 1e-8f);
            batch_gradient = batch_gradient + g;
        }

        Gradient avg_gradient = (1 / (float)BATCH_SIZE) * batch_gradient;

        float batch_entropy_loss = batch_loss / BATCH_SIZE;
        std::cout << std::endl << "loss:" << batch_entropy_loss << std::endl;    

        NN.descend(avg_gradient, STEP_SIZE);
        
    }
        
        
    auto test_images = parseMnistImages("mnist_data/t10k-images.idx3-ubyte");
    auto test_labels = parseMnistLabels("mnist_data/t10k-labels.idx1-ubyte");
    
    int index {0};
    int num_correct {0};
    for (auto& image : test_images)
    {
        Vec<784> input = getInputFromImage(image);
        Vec<10> output = NN.forward(input);

        int output_num = getOutput(output);
        int actual_num = (int)test_labels[index];

        std::cout << "Predicted: " << output_num << " ";
        std::cout << "Actual: " << actual_num << std::endl;

        if (output_num == actual_num) num_correct++;

        ++index;
    }
    std::cout << "Overall Accuracy: " << (float)num_correct / test_images.size();
    
    saveNN("digit_nn.nn", NN);
}