#pragma once

#include "../Tensor/Tensor.hh"
#include <vector>
#include <iostream>
#include <fstream>
#include <array>
#include <stdio.h>

using model_type = float;

enum class set_type
{
    CIFAR_10,
    XOR
};

class DatasetHandler
{
public:
    void read(const char* file, set_type type)
    {
        switch (type)
        {
        case set_type::CIFAR_10:
            load_cifar_10(file);
            break;
        case set_type::XOR:
            load_xor();
            break;
        
        default:
            break;
        }
    }

    /*template <typename MAT_TYPE = float>
    std::vector<Tensor<MAT_TYPE>> normalize(const std::vector<Tensor<model_type>>& set, MAT_TYPE value) const
    {
        std::vector<Tensor<MAT_TYPE>> norm_set;
        for (const Tensor<model_type>& m : set)
        {
            Tensor<MAT_TYPE> mat(m.get_shape().at(0), m.get_shape().at(1));
            for (int y = 0; y < m.get_shape().at(0); ++y)
            {
                for (int x = 0; x < m.get_shape().at(1); ++x)
                    mat(y, x) = m(y, x) / value;
            }
            norm_set.emplace_back(mat);
        }
        return norm_set;
    }

    std::vector<Tensor<model_type>> binarize(void) const
    {
        std::vector<Tensor<model_type>> set;
        for (const Tensor<model_type>& t : x)
        {
            Tensor<model_type> b_w(1, t.get_shape().at(1), t.get_shape().at(2));

            for (int y = 0; y < t.get_shape().at(1); ++y)
            {
                for (int x = 0; x < t.get_shape().at(2); ++x)
                {
                    b_w(0, y, x) = t(0, y, x) * 0.299f
                                    + t(1, y, x) * 0.587f
                                    + t(3, y, x) * 0.114f;
                }
            }

        }
        return set;
    }*/

    std::vector<Tensor<model_type>>& get_training(void)
    {
        return x;
    }

    std::vector<Tensor<model_type>>& get_labels(void)
    {
        return y;
    }

private:
    void load_cifar_10(const char* path)
    {
        static constexpr auto nb_image = 10000;
        static constexpr auto image_width = 32;
        static constexpr auto image_height = 32;
        static constexpr auto channel_size = 3;

        std::ifstream file;
        file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

        if (!file) {
            std::cout << "Error opening file: " << path << std::endl;
            return;
        }

        auto file_size = file.tellg();
        std::unique_ptr<char[]> buffer(new char[file_size]);

        //Read the entire file at once
        file.seekg(0, std::ios::beg);
        file.read(buffer.get(), file_size);
        file.close();

        for (int image = 0; image < nb_image; ++image)
        {
            int offset = image * (image_height * image_width * 3 + 1);
            char value;
            {
                Tensor<model_type> label({10, 1});
                label.fill(fill_type::ZEROS);
                value = buffer[offset];
                label(static_cast<int>(value), 0) = static_cast<model_type>(1);
                y.emplace_back(label);
            }


            Tensor<model_type> rgb({channel_size, image_height, image_width});

            for (int channel = 0; channel < channel_size; ++channel)
            {
                for (int y = 0; y < image_height; ++y)
                {
                    for (int x = 0; x < image_width; ++x)
                    {
                        value = buffer[offset + x + y * image_width];
                        rgb(channel, y, x) = static_cast<model_type>(value);
                    }
                }
            }
            x.emplace_back(rgb);
        }
    }

    auto get_xor(unsigned a, unsigned b)
    {
        auto mx = Tensor<model_type>({2, 1});
        mx(0, 0) = a;
        mx(1, 0) = b;

        auto my = Tensor<model_type>({1, 1});
        my(0, 0) = a ^ b;

        return std::make_pair(mx, my);
    }

    void load_xor(void)
    {
        auto a = get_xor(0, 0);
        x.emplace_back(a.first);
        y.emplace_back(a.second);

        auto b = get_xor(0, 1);
        x.emplace_back(b.first);
        y.emplace_back(b.second);

        auto c = get_xor(1, 0);
        x.emplace_back(c.first);
        y.emplace_back(c.second);

        auto d = get_xor(1, 1);
        x.emplace_back(d.first);
        y.emplace_back(d.second);
    }

    std::ifstream::pos_type filesize(const char* filename)
    {
        std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
        return in.tellg();
    }

    std::vector<Tensor<model_type>> x;
    std::vector<Tensor<model_type>> y;
};