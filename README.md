# prophecy

A deep neural networks framework similar to Keras and written from scratch in C++/CUDA.

## Compilation

1. `source /opt/anaconda/bin/activate root`
2. `sudo conda install -c conda-forge xtensor xtensor-blas openblas lapack xtensor-python pybind11`
3. `mkdir build; cd build`
4. `cmake ..`
5. `make`

## Usage

Open a Python3 interpreter in the build folder and create a simple model to learn the XOR gate.

```python
import prophecy as p
import numpy as np

model = p.Model()

model.add(p.layers.input(2))
model.add(p.layers.dense(2, p.activations.sigmoid()))
model.add(p.layers.dense(1, p.activations.sigmoid()))

# Create dataset
x = np.array([0, 0, 0, 1, 1, 0, 1, 1])
y = np.array([0, 1, 1, 0])
x = x.reshape([4, 2, 1]);
y = y.reshape([4, 1, 1]);
# print(x, x.shape)
# print(y, y.shape)

# Train
model.set_learning_rate(0.1)
model.train(x, y, 1, 10000)

# Evaluate
for x_i in x:
    print("Input", x_i.T)
    print("Output", model.predict(x_i))

```

## To-Do

Refer to [this page](https://github.com/theolepage/prophecy/projects/1).
