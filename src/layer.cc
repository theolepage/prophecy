#include "layer.hh"

Layer::Layer(unsigned nb_neurons)
    : nb_neurons_(nb_neurons)
{}

unsigned Layer::get_nb_neurons()
{
    return nb_neurons_;
}
