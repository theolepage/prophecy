#pragma once

#include "../matrix/matrix.hh"
#include <vector>
#include <iostream>
#include <fstream>
#include <array>
#include <stdio.h>

using data_type = char;

enum class set_type
{
    CIFAR_10
};

using c_training_set = std::vector<std::array<Matrix<data_type>, 3>>;
using c_label_set = std::vector<Matrix<data_type>>;

class Dataset_handler
{
public:
    void read(const char* file, set_type type)
    {
        switch (type)
        {
        case set_type::CIFAR_10:
            load_cifar_10(file);
            break;
        
        default:
            break;
        }
    }

    template <typename MAT_TYPE = float>
    std::vector<Matrix<MAT_TYPE>> normalize(const std::vector<Matrix<data_type>>& set, MAT_TYPE value) const
    {
        std::vector<Matrix<MAT_TYPE>> norm_set;
        for (const Matrix<data_type>& m : set)
        {
            Matrix<MAT_TYPE> mat(m.get_rows(), m.get_cols());
            for (int y = 0; y < m.get_rows(); ++y)
            {
                for (int x = 0; x < m.get_cols(); ++x)
                    mat(y, x) = m(y, x) / value;
            }
            norm_set.emplace_back(mat);
        }
        return norm_set;
    }

    std::vector<Matrix<data_type>> binarize(void) const
    {
        class Functor {
            public:
                Functor(const std::array<Matrix<data_type>, 3>* t) : counter_(0), t_(t)
                {}

                data_type operator()()
                {

                    data_type val = (*t_)[0](counter_) * 0.299f
                    + (*t_)[1](counter_) * 0.587f
                    + (*t_)[2](counter_) * 0.114f;
                    ++counter_;
                    return val;
                }

            private:
                int counter_;
                const std::array<Matrix<data_type>, 3>* t_;
        };

        std::vector<Matrix<data_type>> set;
        for (const std::array<Matrix<data_type>, 3>& t : x)
        {
            Functor func(&t);
            Matrix<data_type> black_white(t[0].get_rows(), t[0].get_cols());
            black_white.fill<Functor>(func);

            set.emplace_back(black_white);
        }
        return set;
    }

    c_label_set& get_labels()
    {
        return y;
    }

private:
    void load_cifar_10(const char* file)
    {
        static constexpr auto nb_image = 10000;
        static constexpr auto image_width = 32;
        static constexpr auto image_height = 32;
        std::ifstream file_reader(file);
        if (!file_reader.is_open())
            throw "bad file";

        std::cout << filesize(file) << std::endl;

        for (int image = 0; image < nb_image; ++image)
        {
            data_type value;
            {
                Matrix<data_type> label(10, 1);
                label.fill(fill_type::ZERO);
                file_reader >> value;
                label(value, 0) = static_cast<data_type>(1);
                y.emplace_back(label);
            }

            
            std::array<Matrix<data_type>, 3> rgb{
                {Matrix<data_type>(32, 32), Matrix<data_type>(32, 32), Matrix<data_type>(32, 32)}
            };

            int a = 0;
            for (int channel = 0; channel < 3; ++channel)
            {
                for (int y = 0; y < image_height; ++y)
                {
                    for (int x = 0; x < image_width; ++x)
                    {
                        a++;
                        file_reader >> value;
                        rgb[channel](y, x) = value;
                    }
                }
            }
            x.emplace_back(rgb);
        }
    }

    std::ifstream::pos_type filesize(const char* filename)
    {
        std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
        return in.tellg();
    }

    c_training_set x;
    c_label_set y;
};