# Experiments of Aggregate Computing combined with AI 

## Hop-count with regression

###Description:

Tn this experiment, I have tried to implement the hop count algorithm using a standard ML regression technique.
To generate the data set, I have run multiple Alchemist simulation in which each node has a global system vision. 
Hence, using a breadth graph search, each node knows the correct hop count value.
After a predefined period, for each node, [ExtractCsv](src/main/scala/it/unibo/alchemist/model/implementations/actions/ExtractCsv.scala) exports:
1. min neighbour value;
2. output value (target);
3. a boolean value that tells if a node is a source or not.

In each simulation (program [HopCountOracle](src/main/scala/it/unibo/simulations/HopCountOracle.scala) configuration [simulation.yml](src/main/yaml/simulation.yml)), 
there are 150 nodes displaced randomly in a square large 500x500. 

The training function has the shape of:

*(min neighbour, source) => output*

In total, I have gathered 900 samples.
####Traning configuration
I have tried to train:
- linear regressor;
- gradient regressor;
- random forest regressor.

Considering the learning algorithm simplicity, I choose the smile framework to validate the result.
####Validation configuration
To validate the result (program [PerformanceComparator](src/main/scala/it/unibo/casestudy/PerformanceComparator.scala) configuration [multi_validation.yml](src/main/yaml/multi_validation.yml) I run standard hop count implementation and another regression model. Here, I have decided to increment the node count, for seeing if the model succeeded in the generalization task. The error is computed as a squared error for each time sample.
####What happens
The liner model learns the function (m + 1) but obviously can't learn **when a source is true the output is always 0**. Gradient and random forest model don't generalize, so with higher node count don't increment the value by one.
So, to validate the result, I introduce a "bias" in the execution, namely:
```
mux (source) { 0 } { regression.predict(input) }.
```
doing so, the error is the following:
![Result](data/imgs/result.png)
####Final remarks
TODO
