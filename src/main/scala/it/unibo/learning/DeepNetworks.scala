package it.unibo.learning

import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.dataset.{DataSet => JDataSet}
import org.nd4j.linalg.lossfunctions.LossFunctions

object DeepNetworks {
  case class Seed(value: Long)
  case class DataSetSplit(trainingSet: JDataSet, validationSet: JDataSet, testSet: JDataSet)

  def multiLayerRegressionConfiguration(
      inputSize: Int,
      outputSize: Int,
      hidden: List[Int]
  )(implicit seed: Seed = Seed(42)): MultiLayerConfiguration = {
    val hiddenLayers = hidden
      .map(new DenseLayer.Builder().units(_))
      .map(_.activation(Activation.RELU))
      .map(_.build())
    val output = new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
      .activation(Activation.IDENTITY)
      .nIn(hidden.reverse.head)
      .nOut(outputSize)
      .build()

    val layers = hiddenLayers ::: output :: Nil
    new NeuralNetConfiguration.Builder() //hyper parameter section
      .seed(seed.value)
      .updater(new Adam())
      .list(layers: _*)
      .backpropType(BackpropType.Standard)
      .setInputType(InputType.feedForward(inputSize))
      .build()
  }
}
