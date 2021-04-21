package it.unibo.learning

import it.unibo.learning.DeepNetworks.DataSetSplit
import it.unibo.learning.DeepNetworks.Seed
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.api.Layer.TrainingMode
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.evaluation.regression.RegressionEvaluation
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator

import java.io.File

object NetworkExport extends App {
  implicit val seed: Seed = Seed(42)
  //network configuration
  private val inputSize  = 2
  private val outputSize = 1
  private val hidden     = 10 :: 5 :: Nil
  //dataset information
  private val regressionIndex = 0
  private val iteration       = 30
  private val epoch           = 100
  private val batch           = 30

  private val network = new MultiLayerNetwork(
    DeepNetworks
      .multiLayerRegressionConfiguration(inputSize, outputSize, hidden)
  )

  private def isProbability(data: Double): Boolean = data >= 0 && data <= 1

  private def prepareDataset(
      file: String,
      splitValidation: Double,
      splitTest: Double,
      batchSize: Int = 1,
      labelRegressionRange: (Int, Int)
  )(implicit seed: Seed): DataSetSplit = {
    require(isProbability(splitValidation) && isProbability(splitTest))
    val reader = new CSVRecordReader(',')
    reader.initialize(new FileSplit(new File(file)))
    val dataSetIterator =
      new RecordReaderDataSetIterator(reader, batchSize, labelRegressionRange._1, labelRegressionRange._2, true)

    val allData = dataSetIterator.next()
    allData.shuffle(seed.value)
    val trainAndTest = allData.splitTestAndTrain(1 - splitTest)
    val (trainAndValidation, test) =
      (trainAndTest.getTrain.splitTestAndTrain(1 - splitValidation), trainAndTest.getTest)
    val (train, validation) = (trainAndValidation.getTrain, trainAndValidation.getTest)
    DataSetSplit(train, validation, test)
  }

  val dataset =
    prepareDataset(
      "output.csv",
      splitValidation = 0.2,
      splitTest = 0.1,
      batchSize = 10000,
      labelRegressionRange = (regressionIndex, regressionIndex)
    )

  network.setIterationCount(iteration)
  network.init()
  network.setListeners(new SimpleScoreLister)
  val iterator = new ListDataSetIterator[DataSet](dataset.trainingSet.asList(), batch)
  network.fit(iterator, epoch)
  val evaluation = new RegressionEvaluation()
  println(dataset.testSet.getLabels)
  evaluation.eval(dataset.testSet.getLabels, network.output(dataset.testSet.getFeatures))
  println(evaluation.stats())
  println(network.feedForward(new NDArray(Array(0.0f, 0.0f))))
  println(network.feedForward(new NDArray(Array(0.0f, 1.0f))))
  println(network.feedForward(new NDArray(Array(0.0f, 10.0f))))
  println(network.feedForward(new NDArray(Array(0.0f, 100.0f))))
}
