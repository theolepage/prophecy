# prophecy

Check this out Google.

## Usage

```cpp
Model model = Model();

// Create model
model.add(make_shared<InputLayer>(2));
model.add(make_shared<DenseLayer>(8, make_shared<SigmoidActivationFunction>()));
model.add(make_shared<DenseLayer>(2, make_shared<SigmoidActivationFunction>()));

// Train model
model.compile(0.01);
model.train(x_train, y_train, 10, 8);

// Create input value
auto x = make_shared<Matrix>(2, 1);
(*x)(0, 0) = 1;
(*x)(1, 0) = 0;

// Make a prediction
auto y = model.predict(x);
cout << *y;
```

## To-Do

### Dense layers

- [ ] Test current framework with a XOR
- [ ] Clean code
- [ ] Write tests
- [ ] Update docs

### Next features

- [ ] Implement Model::load(), Model::save() and Dataset class
- [ ] Implement convolutional layers (and dropout, max pooling, ...)
- [ ] Optimize computations
