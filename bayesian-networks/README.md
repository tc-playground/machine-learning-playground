# Bayesian Networks

## [`Bayesian Networks`](https://towardsdatascience.com/introduction-to-bayesian-networks-81031eeed94e)

* A `Bayesian network` is a `directed acyclic graph` in which each `edge` corresponds to a `conditional dependency`, and each `node` corresponds to a `unique random variable`. 

* `Bayesian networks` are a type of probabilistic graphical model that uses `Bayesian inference` for probability computations.

* `Bayesian networks` aim to `model conditional dependence`, and therefore `causation`, by representing `conditional dependence` by `edges in a directed graph`.

* `Conditional independence` between two random variables, `A` and `B`, given another random variable, `C`:

    * `P(A,B|C) = P(A|C) * P(B|C)`

    * -> `P(A|B,C) = P(A|C)`

* A compact, `factorized representation` of the `joint probability distribution` by taking advantage of `conditional independence`.

