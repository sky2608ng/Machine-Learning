# Chapter2
## Unit step function(Heaviside step function):

>F(x) = if x >= 0, 1; else -1

## Weight Update
>w = w + delta(w)
delta(w) = learning rate * (real_output - predicted output) * input
> y = activation()

## Using Standard Deviation in preprocessing data
>x = (x - x.std) / x.mean

## Gradient Descent
>J(w) = 1/2 * sum((real_output - predicted_output)^2)
w = w + delta(w)
delta(w) = learning_rate * partial_diff(J(w))

## Stochastic Gradient Descent(Better approach)
If the data is large, the Batch Gradient Descent uses a lot of resource to update the weight in each attempts.  
Therefore, Stochastic Gradient Descent could solve the problem and it is more efficient than BGD.  
For precision of SGD, the order of the data is a key, make sure the datas are randomly in-placed, otherwise, the correctness will be affected
>delta(w) = learning rate * sum(real_output - predicted_output) * input  

The learning rate of SGD will gradually decrease over time
>c = c1 / (number of iteration + c2)

## Other Gradient Descent
1. Batch Gradient Descent
2. Small Batch Gradient Descent

