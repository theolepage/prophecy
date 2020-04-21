# To-Do

- [ ] reduce(): make sum generic by passing a function (+, -, *, /)
- [ ] pad(): pad specified dimensions
- [ ] conv(): conv specified dimensions with stride and padding
- [ ] get(): when axis is < shape => return tensor
- [ ] transpose(): allow nd tensor but apply transpose on last 2 dimensions
- [ ] matmul(): allow nd tensor but apply matmul on last 2 dimensions

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
