#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <array>
#include <stdio.h>

#include "tensor/tensor.hh"

using dh_model_type = float;

enum class set_type
{
    CIFAR_10,
    MNIST,
    XOR
};

class DatasetHandler
{
public:
    DatasetHandler()
        : limit_(-1)
    {}

    void read(std::string file, set_type type)
    {
        switch (type)
        {
        case set_type::CIFAR_10:
            load_cifar_10(file);
            break;
        case set_type::MNIST:
            load_mnist(file);
            break;
        case set_type::XOR:
            load_xor();
            break;

        default:
            break;
        }
    }

    template <typename MAT_TYPE = float>
    std::vector<Tensor<MAT_TYPE>> normalize(const std::vector<Tensor<dh_model_type>>& set, MAT_TYPE value) const
    {
        std::vector<Tensor<MAT_TYPE>> norm_set;
        for (const Tensor<dh_model_type>& m : set)
        {
            Tensor<dh_model_type> out(m);
            out /= value;
            norm_set.emplace_back(out);
        }
        return norm_set;
    }

    /*
    std::vector<Tensor<dh_model_type>> binarize(void) const
    {
        std::vector<Tensor<dh_model_type>> set;
        for (const Tensor<dh_model_type>& t : x)
        {
            Tensor<dh_model_type> b_w(1, t.get_shape().at(1), t.get_shape().at(2));

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

    std::vector<Tensor<dh_model_type>>& get_training(void)
    {
        return x;
    }

    std::vector<Tensor<dh_model_type>>& get_labels(void)
    {
        return y;
    }

    long long get_limit(void) const
    {
        return limit_;
    }

    void set_limit(long long limit)
    {
        limit_ = limit;
    }

private:
    void load_cifar_10(std::string path)
    {
        uint nb_image = 10000;
        static constexpr auto image_width = 32;
        static constexpr auto image_height = 32;
        static constexpr auto channel_size = 3;

        std::ifstream file;
        file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

        if (!file)
        {
            std::cout << "Error opening file: " << path << std::endl;
            return;
        }

        auto file_size = file.tellg();
        std::unique_ptr<char[]> buffer(new char[file_size]);

        // Read the entire file at once
        file.seekg(0, std::ios::beg);
        file.read(buffer.get(), file_size);
        file.close();

        // Apply limit
        if (limit_ >= 0 && limit_ < nb_image)
            nb_image = limit_;

        for (uint image = 0; image < nb_image; ++image)
        {
            const uint offset = image * (image_height * image_width * 3 + 1);
            unsigned char value;
            {
                Tensor<dh_model_type> label({ 10, 1 });
                label.fill(fill_type::ZEROS);
                value = buffer[offset];
                label(static_cast<uint>(value), 0u) = static_cast<dh_model_type>(1);
                y.emplace_back(label);
            }


            Tensor<dh_model_type> rgb({ channel_size, image_height, image_width });

            for (uint channel = 0; channel < channel_size; ++channel)
            {
                for (uint y = 0; y < image_height; ++y)
                {
                    for (uint x = 0; x < image_width; ++x)
                    {
                        value = buffer[1 + offset + y * image_width + channel * (image_width * image_height) + x];
                        rgb(channel, y, x) = static_cast<dh_model_type>(value);
                    }
                }
            }
            x.emplace_back(rgb);
        }
    }

    void load_mnist_labels(std::string path)
    {
        path.append("/train-labels-idx1-ubyte");

        uint nb_image = 60000;

        // Open file
        std::ifstream file;
        file.open(path, std::ios::in | std::ios::binary | std::ios::ate);
        if (!file)
        {
            std::cout << "Error opening file: " << path << std::endl;
            return;
        }

        // Read the entire file at once
        auto file_size = file.tellg();
        std::unique_ptr<char[]> buffer(new char[file_size]);
        file.seekg(0, std::ios::beg);
        file.read(buffer.get(), file_size);
        file.close();

        // Apply limit
        if (limit_ >= 0 && limit_ < nb_image)
            nb_image = limit_;

        // Labels
        for (uint image = 0; image < nb_image; image++)
        {
            Tensor<dh_model_type> label({ 10, 1 });
            label.fill(fill_type::ZEROS);

            const uint value = static_cast<uint>(buffer[8 + image]);

            label({ value, 0 }) = static_cast<dh_model_type>(1);
            y.emplace_back(label);
        }
    }

    void load_mnist_images(std::string path)
    {
        path.append("/train-images-idx3-ubyte");

        uint nb_image = 60000;
        static constexpr auto image_width = 28;
        static constexpr auto image_height = 28;

        // Open file
        std::ifstream file;
        file.open(path, std::ios::in | std::ios::binary | std::ios::ate);
        if (!file)
        {
            std::cout << "Error opening file: " << path << std::endl;
            return;
        }

        // Read the entire file at once
        auto file_size = file.tellg();
        std::unique_ptr<char[]> buffer(new char[file_size]);
        file.seekg(0, std::ios::beg);
        file.read(buffer.get(), file_size);
        file.close();

        // Apply limit
        if (limit_ >= 0 && limit_ < nb_image)
            nb_image = limit_;

        // Images
        for (uint image = 0; image < nb_image; ++image)
        {
            int offset = 16 + image * (image_height * image_width);
            Tensor<dh_model_type> rgb({ 1, image_height, image_width });

            for (uint y = 0; y < image_height; ++y)
            {
                for (uint x = 0; x < image_width; ++x)
                {
                    const uint value = static_cast<uint>(buffer[offset + y * image_width + x]);
                    rgb({ 0, y, x }) = static_cast<dh_model_type>(value / 255.0f);
                }
            }
            x.emplace_back(rgb);
        }
    }

    void load_mnist(std::string path)
    {
        load_mnist_labels(path);
        load_mnist_images(path);
    }

    auto get_xor(const uint a, const uint b)
    {
        auto mx = Tensor<dh_model_type>({2u, 1u});
        mx(0u, 0u) = a;
        mx(1u, 0u) = b;

        auto my = Tensor<dh_model_type>({1u, 1u});
        my(0u, 0u) = a ^ b;

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

    std::vector<Tensor<dh_model_type>> x;
    std::vector<Tensor<dh_model_type>> y;
    long long limit_;
};
