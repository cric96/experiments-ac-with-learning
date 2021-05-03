package it.unibo.learning

import it.unibo.learning.DeepNetworks.DataSetSplit
import it.unibo.learning.DeepNetworks.Seed
import org.apache.commons.io.FilenameUtils
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.nn.conf.layers.PoolingType
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.model.stats.StatsListener
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.evaluation.regression.RegressionEvaluation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

import java.io.File

object NetworkExport {
  implicit val seed: Seed = Seed(42)
  //network configuration
  private val outputSize = 1

  private val hidden = {
    import DeepNetworks._
    conv1d(3, 1, 16) ::
      conv1d(1, 1, 8) ::
      Nil
  }
  //dataset information
  private val regressionIndex = 0
  private val epoch           = 1000
  private val patience        = 10
  private val batchSize       = 1000
  private val splitValidation = 0.2
  private val splitTest       = 0.1

  //dataset preparation
  private val dataset =
    prepareDataset(
      "output.csv",
      splitValidation = splitValidation,
      splitTest = splitTest,
      batchSize = batchSize,
      labelRegressionRange = (regressionIndex, regressionIndex)
    )
  private val trainIterator      = DeepNetworks.wrapDataSetToIterator(dataset.trainingSet, 1) //SGD... we can improve..
  private val validationIterator = DeepNetworks.wrapDataSetToIterator(dataset.validationSet, 1)

  //neural network
  private val configuration = DeepNetworks.fullyConvolutionalNetwork1D(hidden, outputSize, PoolingType.AVG)
  private val network       = new MultiLayerNetwork(configuration)

  def main(args: Array[String]): Unit = {
    network.init()
    //network initialization
    network.init()
    attachUIServer(network)
    //train
    val trainer = configureTrainer(network, epoch, patience, validationIterator, trainIterator)
    trainer.fit()
    //evaluation
    val evaluation = new RegressionEvaluation()
    evaluation.eval(dataset.testSet.getLabels, network.output(dataset.testSet.getFeatures))
    println(evaluation.stats())
    //store
    ModelSerializer.writeModel(network, "src/main/resources/network", false)
  }

  //utility functions
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

  private def attachUIServer(network: MultiLayerNetwork): Unit = {
    //gui configuration
    val uiServer     = UIServer.getInstance()
    val statsStorage = new InMemoryStatsStorage
    uiServer.attach(statsStorage)
    //add listener
    network.setListeners(new SimpleScoreLister, new StatsListener(statsStorage))
  }

  private def configureTrainer(
      network: MultiLayerNetwork,
      epochCount: Int,
      patience: Int,
      validationSet: DataSetIterator,
      trainingSet: DataSetIterator
  ): EarlyStoppingTrainer = {
    val modelSaver = configureModelSaver
    //early stopping configuration
    val esConf = new EarlyStoppingConfiguration.Builder()
      .epochTerminationConditions(
        new MaxEpochsTerminationCondition(epochCount),
        new ScoreImprovementEpochTerminationCondition(patience)
      )
      .scoreCalculator(new DataSetLossCalculator(validationSet, true))
      .evaluateEveryNEpochs(1)
      .modelSaver(modelSaver)
      .build()
    new EarlyStoppingTrainer(esConf, network, trainingSet)
  }

  private def configureModelSaver: EarlyStoppingModelSaver[MultiLayerNetwork] = {
    //in-memory directory
    val tempDir: String          = System.getProperty("java.io.tmpdir")
    val exampleDirectory: String = FilenameUtils.concat(tempDir, "DL4JEarlyStoppingExample/")
    val dirFile: File            = new File(exampleDirectory)
    dirFile.mkdir()
    //model saver for early stopping cycle
    new LocalFileModelSaver(exampleDirectory)
  }
}
