## Structure

### `Layer`

Attributes:
- `number_of_neurons`

### `InputLayer` (extends `Layer`)

### `DenseLayer` (extends `Layer`)

Attributes:
- `weights`
- `biases`

### `Model`

Attributes:
- `layers`

Methods:
- `Model()`
- `add(Layer)`
- `predict(x)`
- `train(x, y, epochs, batch_size, learning_rate)`
- `evaluate(x, y)`
- `summary()`
- `save(path)`
- `load(path)`

### Matrix

Attributes:
- `rows`
- `cols`
- `data`

Methods:
- `Matrix(rows, cols)`
- `operator()(row, col)`
- `operator+(m1, m2)`
- `operator-(m1, m2)`
- `operator*(m1, m2)`
- `fill_with_random()`
- `transpose()`
