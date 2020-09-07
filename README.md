Two experiments on HIGGS dataset
================================


Preliminaries
--------------

During the reading quite interesting paper about whole-dataset neural network optimization [Training Neural Networks Without Gradients:A Scalable ADMM Approach](https://arxiv.org/pdf/1605.02026.pdf) I found an interesting phenomena there: for some relatively simple dataset it was not possible for stochastic gradient descent to compete with whole-dataset optimization algorithm. Go to page 8 for figures.

In the paper SGD stops at about 55% accuracy while other methods (ADMM and conjugate gradient) go up to 64%. The dataset is [HIGGS dataset](https://archive.ics.uci.edu/ml/datasets/HIGGS#) ([paper](https://arxiv.org/pdf/1402.4735.pdf))

I have some reservation about SGD because it is quite complex to implement properly. What learning rate to choose? What batch size is best? How to efficiently sample dataset? Etc. Thus here I will not implement SGD version and instead I will rely on the results from ADMM paper.

And when I had free lazy Sunday I decided to experiment on that dataset.


First experiment - use logistic regression
------------------------------------------

The "neural network" implemented in the a.hs module is, basically, a logistic regression. The optimization algorithm is Iterated Reweighted Least Squares as I currently understand that.

Initially I wanted to implement [Gated Linear Networks](https://arxiv.org/pdf/1910.01526.pdf). But logistic regression is part of that implementation so I decided to start from it and go to full GLNs if nedded.

From the get go I had accuracy of 64% after about 8 iterations. On a part of dataset, because I needed results quickly, but data there is of same nature and source (not a case with Penn Digits, BTW).

So, first surprise: 29 coefficients optimized with whole-dataset optimization algorithm seem to be as good as several thousands coefficients in real neural network.


Second experiment - add second degree information
------------------------------------------------

An idea I long longed to try out: is it possible to use polynomials in the function approximation? The idea explained in the [Polynomial Regression as an Alternative to Neural Nets](https://arxiv.org/pdf/1806.06850.pdf), but I didn't want to go that far. A single activation with full power of polynomials of second degree needs O(N^2) coefficients (N is the size of input). I needed something that is less extreme.

Why not add second degree information with diagonal matrix? E.g., complete polynomial has form ``x^TAx+b^Tx+c`` and ``A`` there is full NxN matrix. But we can, say, split it into diagonal part ``D`` and general square matrix ``R`` and throw away ``R`` unless we know how to deal with it. The resulting "polynomial" will have form ``x^TDx + b^Tx + c``.

Basically, we invented some features here and these features are squared original inputs.

And with that the accuracy rises to 66% on held out test data. Double coefficients size, two percents of accuracy gained.


Haskell's woes
------------

I spent a good two thirds of experimentation time fighting with HMatrix and laziness. I had to implement my own CSV reading because simpler way wanted to take 300GB of RAM (predicted from smaller parts) for 7G CSV dataset with 320M doubles.

I also do not fancy cabal files, they take fun away (v2 style). Thus, here's README and just one source file which I like better.

Final thoughts
--------------

I really hope these experiments will be useful for someone else than me. I had my fun, now your turn.

