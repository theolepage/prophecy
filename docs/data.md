# To-Do

- [ ] get(): when axis is < shape => return tensor
- [ ] pad(): pad specified dimensions
- [ ] append()

- [ ] Fusion with Matrix and adapt DenseLayer

# Our data structures

## Conv2DLayer

- list of filters

### Filter

- bias
- list of Matrix

- conv()
- padding() => padding on each Matrix

## Matrix

- sum() = reduce()

# Backpropagation

// db
for (int f = 0; f < filters_.size(); f++)
    filters_[f].db = delta.matrices[f].sum();

// dw
xp = x.padding()
for (int f = 0; f < filters_.size(); f++)
    filters_[f].dw = xp.conv(delta);

// dx
dxp = dx.padding()
dxp = delta.conv(w);
dx = remove padding from dxp

call backprop on prev by passing dx
