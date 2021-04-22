package it.unibo.learning

import it.unibo.learning.DeepNetworks.DataSetSplit
import it.unibo.learning.DeepNetworks.Seed
import org.apache.commons.io.FilenameUtils
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.evaluation.regression.RegressionEvaluation
import org.nd4j.linalg.cpu.nativecpu.NDArray

import java.io.File
import java.util.concurrent.TimeUnit

object NetworkExport extends App {
  implicit val seed: Seed = Seed(42)
  //network configuration
  private val inputSize  = 2
  private val outputSize = 1
  private val hidden     = 8 :: 6 :: 4 :: 2 :: Nil
  //dataset information
  private val regressionIndex = 0
  private val epoch           = 1000
  private val patience        = 10
  private val batchSize       = 1000

  //neural network
  val configuration = DeepNetworks
    .multiLayerRegressionConfiguration(inputSize, outputSize, hidden)
  private val network = new MultiLayerNetwork(configuration)

  private def isProbability(data: Double): Boolean = data >= 0 && data <= 1

  private def prepareDataset(
      file: String,
      splitValidation: Double,
      splitTest: Double,
      batchSize: Int = 2,
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
      batchSize = batchSize,
      labelRegressionRange = (regressionIndex, regressionIndex)
    )
  val trainIterator      = DeepNetworks.wrapDataSetToIterator(dataset.trainingSet)
  val validationIterator = DeepNetworks.wrapDataSetToIterator(dataset.validationSet)
  network.init()
  network.setListeners(new SimpleScoreLister)
  //in-memory directory
  val tempDir: String          = System.getProperty("java.io.tmpdir")
  val exampleDirectory: String = FilenameUtils.concat(tempDir, "DL4JEarlyStoppingExample/")
  val dirFile: File            = new File(exampleDirectory)
  dirFile.mkdir()
  //model saver for early stopping cycle
  val saver = new LocalFileModelSaver(exampleDirectory)

  val esConf = new EarlyStoppingConfiguration.Builder()
    .epochTerminationConditions(
      new MaxEpochsTerminationCondition(epoch),
      new ScoreImprovementEpochTerminationCondition(patience)
    )
    .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(5, TimeUnit.MINUTES))
    .scoreCalculator(new DataSetLossCalculator(validationIterator, true))
    .evaluateEveryNEpochs(1)
    .modelSaver(saver)
    .build()

  val trainer = new EarlyStoppingTrainer(esConf, network, trainIterator)
  trainer.fit()
  val evaluation = new RegressionEvaluation()
  evaluation.eval(dataset.testSet.getLabels, network.output(dataset.testSet.getFeatures))
  println(evaluation.stats())
  println(network.feedForward(new NDArray(Array(0.0f, 0.0f))))
  println(network.feedForward(new NDArray(Array(0.0f, 1.0f))))
  println(network.feedForward(new NDArray(Array(0.0f, 10.0f))))
  println(network.feedForward(new NDArray(Array(0.0f, 100.0f))))
  println(network.feedForward(new NDArray(Array(0.0f, 1000.0f))))
  println(network.feedForward(new NDArray(Array(0.0f, 5000.0f))))
}
