package it.unibo.learning

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.{DataSet => JDataSet}
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.jdk.CollectionConverters.SeqHasAsJava

object DeepNetworks {
  private val defaultSeed = Seed(42)
  //utility case class
  case class Seed(value: Long)
  case class DataSetSplit(trainingSet: DataSetIterator, validationSet: DataSetIterator, testSet: DataSetIterator)
  case class Conv1DLayerInfo(kernelSize: Int, depth: Int, filters: Int)
  def conv1d(kernel: Int, depth: Int, filters: Int): Conv1DLayerInfo = Conv1DLayerInfo(kernel, depth, filters)

  //function to create a multi layer network with regression task
  def multiLayerRegressionConfiguration(
      inputSize: Int,
      outputSize: Int,
      hidden: List[Int]
  )(implicit seed: Seed = defaultSeed): MultiLayerConfiguration = {
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
      .setInputType(InputType.feedForward(inputSize))
      .backpropType(BackpropType.Standard)
      .build()
  }

  def fullyConvolutionalNetwork1D(layers: List[Conv1DLayerInfo], output: Int, poolingType: PoolingType)(implicit
      seed: Seed = defaultSeed
  ): MultiLayerConfiguration = {
    val hidden =
      layers.map(info => new Convolution1DLayer.Builder(info.kernelSize).nIn(info.depth).nOut(info.filters).build())

    val globalAveragePooling = new GlobalPoolingLayer.Builder(poolingType)
      .build()

    val outputLayer = new OutputLayer.Builder()
      .nIn(layers.reverse.head.filters)
      .nOut(output)
      .activation(Activation.IDENTITY)
      .lossFunction(LossFunctions.LossFunction.MSE)
      .build()

    val layersBuilt: List[Layer] = hidden ::: globalAveragePooling :: outputLayer :: Nil
    new NeuralNetConfiguration.Builder()
      .weightInit(WeightInit.RELU)
      .activation(Activation.RELU)
      .convolutionMode(ConvolutionMode.Causal)
      .seed(seed.value)
      .updater(new Adam())
      .list(layersBuilt: _*)
      .backpropType(BackpropType.Standard)
      .build()
  }

  def wrapDataSetToIterator(dataset: List[JDataSet], batchSize: Int = 32): DataSetIterator =
    new ListDataSetIterator(dataset.asJava, batchSize)
}
