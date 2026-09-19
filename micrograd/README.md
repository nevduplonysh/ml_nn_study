#### micrograd

Here I want to briefly describe why and how it works.
<span style="color:gray">For now, without referring to the code.</span>

I have made this small study project, almost fully relying on the Andrej Karpathy's video:
https://youtu.be/VMj-3S1tku0?si=JBnopisGI22K1qhl

My goal was to understand the basic concepts of how gradient calculation is processed and how it can be used to create a simple Neural Net that can solve a binary classification problem.

Micrograd is a small version of autograd. It calculates the gradient for each node in a computational graph.

A gradient is a vector that shows the direction of the steepest increase of a function, and its magnitude shows how fast the function's value changes in that direction. 
 
*-- Why do we want to know it?
-- Because it helps us decrease the value of the loss function.

<span style="color:gray">Here I'm not going to explain what a loss fuction is or how a NN works at all. Maybe I will do it later in main README file.</span>

All calculations start with a loss function at the very end of our computational graph. We say that the value of its grad is 1.
For more clarity, let's look at an example:

Let's say that we have 3 variables:
```
	a = 2
	b = 3
	c = 10
	
	# then:
	
	L = a*b + c
	d = a*b
	L = d + c
```
In the example:
	 - L is our loss function
	 - d is a new variable equal to a\*b

As we said before ```L.grad = 1```, then according to a chain rule: ```dL/dc = dL/dL * dL/dc``` 
And as we can know from our calculus classes: ```dL/dL == 1 and dL/dc == 1```, so ```c.grad == 1 and d.grad == 1```.

*-- Why do you use the chain rule in this simple computing?
-- For more clarity in the next step.*

For now we know how *c* and *d* affect *L*. 
The next step is to calculate how *a* and *b* affect *L* through *d*.
So:
```
	dL/da = dL/dd * dd/da
	
	# it's simple to notice that we already know that dL/dd == 1
	
	dd/da = (a*b)' = b
	
	# then:
	
	dL/da = b and so on dL/db = a
	
	# and 
	
	a.grad = b*1.0 and b.grad = a*1.0 
```
What does it mean? -- It means that at every step we can calculate the current gradient if we know the gradient of the child node in our graph.
So this is why I said that ```L.grad = 1``` and used chain rule for the first step. You can think about it like the starting point in our computation.

This process that we have done manually is called backpropagation.
And it can be done for any equation with almost any complexity (as long as it only has + and * operations) using the chain rule method.

It is possible because of the very simple differentiation rules for + and \*:
	\+
	the gradient is equal to the child's gradient
	\*
	the gradient is equal to the product of child's gradient and the other factor's value 
Thus, we can do this at every iteration of our computation.

This is all about the calculus part. For now I would like to talk a bit more about the logic.

Since we've done this, every value in the graph has a .grad parameter. As I said at the beginning, the gradient helps us decrease the value of the loss function. 
And we need to do this because the lower it is, the more accuracy we have.

Let's return to our example:
```
	a = 2
	a.grad = 3
	b = 3
	b.grad = 2
	c = 10
	c.grad = 1
	d = a*b = 6
	d.grad = 1
	L = d + c = a*b + c = 16
	L.grad = 1
	
	# Positive gradient values mean that if we slightly increase any of these
	# values, the magnitude of L will also increase.
	
	# If a = a + 0.1 * a.grad, then L is equal to 16.9.
	
	# On the other hand, if we slightly decrease any of these values, the 
	# magnitude of L will also decrease.
	
	# If a = a - 0.1 * a.grad, the L is equal to 15.1.
```

As a consequence of the phrase "the direction of the steepest increase of a function" and the example right above, if we want to decrease the value of our loss function, we need to subtract the value of a parameter by the product of its gradient and a learning rate step.

<span style="color:grey">well, that's all for now. goodnight, clever children</span>
