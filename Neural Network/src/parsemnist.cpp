#include "../include/parsemnist.hpp"

uint32_t readInt(std::ifstream& f)
{
    std::array<uint8_t, 4> bytes;
    f.read(reinterpret_cast<char*>(bytes.data()), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

digit_pixels readImage(std::ifstream& f, int num_rows, int num_cols)
{
    digit_pixels image;

    for (int i = 0; i < num_rows; ++i)
    {
        f.read(reinterpret_cast<char*>(image[i].data()), num_cols);
    }

    return image;
}

std::vector<digit_pixels> parseMnistImages(const std::string& filename)
{
    std::ifstream f(filename, std::ios::in | std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Error: could not read file.");

    int magic = readInt(f);
    int num_images = readInt(f);
    int num_rows = readInt(f);
    int num_cols = readInt(f);

    if (magic != 2051) throw std::runtime_error("Error: invalid magic number.");

    std::vector<digit_pixels> images;

    for (int i = 0; i < num_images; ++i)
    {
        images.push_back(readImage(f, num_rows, num_cols));
    }

    f.close();

    return images;
}

std::vector<uint8_t> parseMnistLabels(const std::string& filename)
{
    std::ifstream f(filename, std::ios::in | std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Error: could not read file.");

    int magic = readInt(f);
    int num_labels = readInt(f);
    
    if (magic != 2049) throw std::runtime_error("Error: Invalid magic number.");

    std::vector<uint8_t> labels(num_labels);
    f.read(reinterpret_cast<char*>(labels.data()), num_labels);

    f.close();
    
    return labels;
}
