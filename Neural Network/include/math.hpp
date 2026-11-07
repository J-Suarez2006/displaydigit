#ifndef MATH_HPP
#define MATH_HPP

#include <iostream>
#include <vector>
#include <initializer_list>
#include <iomanip>
#include <cmath>
#include <memory>

#pragma pack(push, 1)
template <int r, int c>
class Mat
{
    public:
    Mat() : data(r * c, 0.0f) {}

    Mat(std::initializer_list<float> values) : data(values)
    {
        if (values.size() != r * c)
        {
            throw std::runtime_error("Error: Initializer list does not match Matrix dimensions!");
        }
    }

    Mat(std::initializer_list<std::initializer_list<float>> rows) 
    {
        if (rows.size() != r)
            throw std::runtime_error("Error: Row count does not match Matrix rows");

        data.reserve(r * c);
        int i = 0;
        for (auto& row : rows)
        {
            if (row.size() != c) 
                throw std::runtime_error("Error: Column count does not match matrix columns");

            data.insert(data.end(), row.begin(), row.end());
            ++i;
        }
    }

    inline float& operator()(int row, int col) 
    {
        if (
            row < 0 || row >= r 
            || col < 0 || col >= c
        )
        {
            throw std::runtime_error("Error: Matrix index out of bounds!");
        }
        return data[row * c + col];
    }

    inline const float& operator()(int row, int col) const
    {
        if (
            row < 0 || row >= r 
            || col < 0 || col >= c
        )
        {
            throw std::runtime_error("Error: Matrix index out of bounds!");
        }
        return data[row * c + col];
    }

    private:
    std::vector<float> data;
};
#pragma pack(pop)

template <int r, int c>
inline std::ostream& operator<<(std::ostream& os, const Mat<r, c>& m)
{
    for (int i = 0; i < r; ++i) {
        os << "[ ";
        for (int j = 0; j < c; ++j) {
            os << m(i, j);
            if (j < c - 1) os << " ";
        }
        os << " ]";
        if (i < r - 1) os << std::endl;
    }

    return os;
}

template <int r, int c>
inline Mat<r, c> operator+(const Mat<r, c>& m1, const Mat<r, c>& m2)
{
    Mat<r, c> res;
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            res(i, j) = m1(i, j) + m2(i, j);

    return res;
}

template <int r, int c>
inline Mat<r, c> operator*(float scalar, const Mat<r, c>& m)
{
    Mat<r, c> res;
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            res(i, j) = scalar * m(i, j);

    return res;
}

template <int r, int c>
inline Mat<r, c> elementwiseMult(const Mat<r, c>& m1, const Mat<r, c>& m2)
{
    Mat<r, c> res;
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            res(i, j) = m1(i, j) * m2(i, j);

    return res;
}

template <int r, int c>
inline Mat<r, c> operator-(const Mat<r, c>& m1, const Mat<r, c>& m2)
{
    return m1 + (-1 * m2);
}

template <int r1, int c1, int r2, int c2>
inline Mat<r1, c2> operator*(const Mat<r1, c1>& m1, const Mat<r2, c2>& m2)
{
    if (c1 != r2)
    {
        throw std::runtime_error("Error: Undefined matrix multiplication operation!");
    }

    int shared = c1;
    Mat<r1, c2> res;

    for (int i = 0; i < r1; ++i)
        for (int j = 0; j < c2; ++j)
            for (int k = 0; k < shared; ++k)
                res(i, j) += m1(i, k) * m2(k, j);
            
    return res;
}

template <int r, int c>
inline Mat<c, r> transpose(const Mat<r, c>& m)
{
    Mat<c, r> res;

    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            res(j, i) = m(i, j);
    
    return res;
}

template <int n>
using Vec = Mat<n, 1>;
// Vectors are treated as column vectors

#endif