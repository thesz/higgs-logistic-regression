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


How to run
----------

```$ ghc -o a-hs --make -odir objs -hidir objs -O3 a.hs && time ./a-hs                        
[1 of 1] Compiling Main             ( a.hs, objs/Main.o )

a.hs:31:1: warning: [-Wtabs]
    Tab character found here, and in 226 further locations.
    Please use spaces instead.
   |
31 |         | otherwise = loop (b - (a LA.#> b)) b
   | ^^^^^^^^
Linking a-hs ...
read 0
read 100000
read 200000 
......
read 10800000                                                                                                                          
read 10900000
read whole 11000000
evaluated: 5566682 from 10500000, ratio 0.5301601904761905, loss 1.0437739473273728
evaluated: 5754537 from 10500000, ratio 0.5480511428571428, loss 0.9885380144308733
evaluated: 6154574 from 10500000, ratio 0.5861499047619048, loss 0.9556088675063659
evaluated: 6457140 from 10500000, ratio 0.6149657142857143, loss 0.9356814631349517
evaluated: 6644622 from 10500000, ratio 0.6328211428571429, loss 0.9232578403532684
evaluated: 6753128 from 10500000, ratio 0.6431550476190476, loss 0.9152138811679993
evaluated: 6818100 from 10500000, ratio 0.6493428571428571, loss 0.9098081830748959
evaluated: 6860362 from 10500000, ratio 0.6533678095238096, loss 0.9060838279450868
evaluated: 6890397 from 10500000, ratio 0.6562282857142857, loss 0.9035132682204905
evaluated: 6912001 from 10500000, ratio 0.6582858095238096, loss 0.9017867699793111
evaluated: 6928849 from 10500000, ratio 0.659890380952381, loss 0.9007185683625898
evaluated: 6943233 from 10500000, ratio 0.6612602857142857, loss 0.9001845304041879
evaluated: 6955524 from 10500000, ratio 0.6624308571428571, loss 0.9000847157095576
evaluated: 6966177 from 10500000, ratio 0.6634454285714285, loss 0.9003347933989551
evaluated: 331634 from 500000, ratio 0.663268, loss 0.9009315807859143
                                 
real    58m48.767s
user    58m25.705s
sys     0m22.231s
sz@Ubuntu-2004-focal-64-minimal:~/play/1/gln$
```

Final thoughts
--------------

I really hope these experiments will be useful for someone else than me. I had my fun, now your turn.

