# Experiments of Aggregate Computing combined with AI 

This repository contains experiments in which we explore the combination of declarative collective programming approach (Aggregate Computing)
with AI techniques.

Each experiment is self-contained in a branch. The branch name tries to clarify the experiment extent (e.g. gradient-regression).

Each branch that ends with *primer* contains the basic configuration for a certain technology/framework

In the main branch, I try to maintain the list (ordered by date) of experiments done with their status/important things discovered.

Here, each experiment is synthesis with:
- the name that link to the branch
- a brief description of the experiment
- a sentence that describes the goal
- a status that could be in :black_circle: Not started :red_circle: Doing, :yellow_circle: On validation (i.e. wait the validation from @metaphori and @mviroli) and :green_circle: Done (the goal is achieved and is validated by the group)
- things discovered with a *qualitative* description, and a color (:black_circle: -> nothing special, ❓ -> some doubts in the approach, :yellow_circle: -> something interesting, :green_circle: -> valuable result / usable in a scientific article)
---
## [Hop count using a temporal window](https://github.com/cric96/experiments-ac-with-learning/tree/???)
**TODO**
### *Goal*
**TODO**
### :black_circle: Status
Not started
### :black_circle: Things discovered
---
## [Hop count using with variable input](https://github.com/cric96/experiments-ac-with-learning/tree/hop-count-variable-input)
Here I try to train some models that are input invariant. I make this choice because in AC we usually have a variable neighbourhood. In particular, I choose one dimensional (Convolutional Neural Network)[https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks] and (Recurrent Neural Network)[https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks].
### *Goal*
Find a model that can be used in our experiment without preprocess (or just a little preprocessing) neighbour values.
### 🔴 Status
Doing
### ❔ Things discovered
In this (commit)[https://github.com/cric96/experiments-ac-with-learning/tree/f36416405f1f85b6b366356ce424dc17c6d27797] there is a first attempt of using AC with RNN (or CNN). Since these models are used in general in time series, I think that they learn some temporal correlation that doesn't make sense in this context (i.e. a *set* of neighbour values). These models could be used in *Hop count using a temporal window* instead.
---
## [Hop count with multiple input](https://github.com/cric96/experiments-ac-with-learning/tree/hop-count-multiple-input)
This experiment is an extension of [Hop count regression](https://github.com/cric96/experiments-ac-with-learning/tree/hop-count-regression) in which we consider a set of values (min, max and average) to compute the hop count.
### *Goal*
Add complexity in order to verify if our approach can be used in a more complex scenario.
### :yellow_circle: Status
On validation
### :black_circle: Things discovered
Nothing special. Nothing special. Please refer to the [README](https://github.com/cric96/experiments-ac-with-learning/tree/hop-count-multiple-input#readme) to gain more information.

---
## [Hop count regression](https://github.com/cric96/experiments-ac-with-learning/tree/hop-count-regression)

This branch contains the experiment in which I try to learn the hop count function using a standard regressor. 
This is the first attempt to mix AC with standard ML technique, so it is only a PoF.
As a simplification, we consider as input only the minimum value within the neighbour.
### *Goal*: 
Take confidence with AC + ML creating a pipeline that will be reused in other examples.
### :green_circle: Status 
Done.
### :black_circle: Things discovered
Nothing special. Please refer to the [README](https://github.com/cric96/experiments-ac-with-learning/tree/hop-count-regression#readme) in the hop count branch.
