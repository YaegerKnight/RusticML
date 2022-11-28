from layers import *
from optimizers import StochasticGradientDescent
from sklearn.datasets import make_regression

X = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
x = [2.0, 3.0, -1.0]
y = [1.0, -1.0, -1.0, 1.0] # desired targets

nn = MLP(3, [4,4,1])
ypred = [nn(x)[0] for x in X]

print(len(ypred))

for k in range(20):

    ypred = [nn(x)[0] for x in X]
    optimizer = StochasticGradientDescent(params= nn.parameters(), lr=0.1)
    # loss = sum((y1 -_y)**2 for y1, _y in zip(ypred, y))
    loss = 0
    for _y, y1 in zip(y, ypred):
        res = (y1 - _y)**2
        loss += res
    
    nn.gradients_to_zero()

    loss.backward()
    optimizer.optimize()

    print(k, loss)

ypred = [nn(x)[0] for x in X]
print(ypred)

nn.save(filename="m1.pkl")

m1_model = MLP.load(filename="m1.pkl")