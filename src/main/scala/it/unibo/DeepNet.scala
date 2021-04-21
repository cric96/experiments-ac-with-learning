package it.unibo

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam

object DeepNet {

  def apply(inputSize: Int, outputSize: Int, hiddenLayersShape: List[Int], seed: Long = 42): MultiLayerNetwork = {
    val layers = (hiddenLayersShape ::: outputSize :: Nil)
      .map(new DenseLayer.Builder().units(_))
      .map(_.activation(Activation.RELU))
      .map(_.build())

    val config = new NeuralNetConfiguration.Builder() //hyper parameter section
      .seed(seed)
      .updater(new Adam())
      .list(layers: _*)
      .setInputType(InputType.feedForward(inputSize))
      .build()
    new MultiLayerNetwork(config)
  }
}
