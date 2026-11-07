#ifndef PARSE_HPP
#define PARSE_HPP

#include <fstream>
#include <vector>
#include <cstdint>
#include <array>

using digit_pixels = std::array<std::array<uint8_t, 28>, 28>;

uint32_t readInt(std::ifstream& f);
digit_pixels readImage(std::ifstream& f, int num_rows, int num_cols);
std::vector<digit_pixels> parseMnistImages(const std::string& filename);
std::vector<uint8_t> parseMnistLabels(const std::string& filename);

#endif