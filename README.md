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
- a status that could be in :red_circle: Doing, :yellow_circle: On Validation (i.e. wait the validation from @metaphori and @mviroli) and :green_circle: Done (the goal is achieved and is validated by the group)
- things discovered with a *qualitative* description, and a color (:black_circle: -> nothing special, :yellow_circle: -> something interesting, :green_circle: -> valuable result / usable in a scientific article)
---
## [Hop count with multiple input](https://github.com/cric96/experiments-ac-with-learning/tree/hop-count-multiple-input)
**TODO**
### *Goal*
**TODO**
### :red_circle: Status
Doing.
### :black_circle: Things discovered
**TODO**

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