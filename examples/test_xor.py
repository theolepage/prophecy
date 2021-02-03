import prophecy as p
import numpy as np

# Create dataset
x = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape([4, 2])
y = np.array([0, 1, 1, 0]).reshape([4, 1])

# Create model
model = p.Model()
model.add(p.layers.input(2))
model.add(p.layers.dense(2, p.activations.sigmoid()))
model.add(p.layers.dense(1, p.activations.sigmoid()))
model.summary()

# Train
print()
model.set_learning_rate(0.25)
model.train(x, y, 4, 10000)

# Evaluate
print()
for x_i in x:
    print(x_i.T, "=>", model.predict(x_i))
print("Total loss:", model.evaluate(x, y))