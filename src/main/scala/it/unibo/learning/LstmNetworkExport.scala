package it.unibo.learning

import it.unibo.learning.DeepNetworks.DataSetSplit
import it.unibo.learning.DeepNetworks.Seed
import it.unibo.learning.DeepNetworks.wrapDataSetToSequenceIterator
import org.apache.commons.io.FilenameUtils
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.writable.Writable
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.model.stats.StatsListener
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.evaluation.regression.RegressionEvaluation
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import DeepNetworks._
import java.io.File
import scala.jdk.CollectionConverters.IteratorHasAsScala
import scala.jdk.CollectionConverters.ListHasAsScala
import scala.util.Random

object LstmNetworkExport {
  implicit val seed: Seed = Seed(42)
  val random              = new Random(seed.value)
  //network configuration
  private val outputSize = 1
  //dataset information
  private val epoch           = 10000
  private val patience        = 5
  private val splitValidation = 0.2
  private val splitTest       = 0.1
  private val batchSize       = 1

  //layers
  private val hiddenLayer = {
    lstm(1, 20) :: Nil
  }

  //dataset preparation
  private val dataset =
    prepareDataset(
      "output.csv",
      splitValidation = splitValidation,
      splitTest = splitTest
    )
  //neural network
  private val lstmConfiguration = DeepNetworks.lstmRecurrentNetwork(hiddenLayer, outputSize)
  private val lstmNetwork       = new MultiLayerNetwork(lstmConfiguration)

  def main(args: Array[String]): Unit = {
    //network initialization
    lstmNetwork.init()
    attachUIServer(lstmNetwork)
    lstmNetwork.addListeners(new SimpleScoreListener)
    //train
    val trainer = configureTrainer(lstmNetwork, epoch, patience, dataset.validationSet, dataset.trainingSet)
    trainer.fit()
    //evaluation
    val evaluation   = new RegressionEvaluation()
    val testElements = dataset.testSet.asScala.toList
    val labels       = testElements.map(_.getLabels)
    val feature      = testElements.map(_.getFeatures)
    feature.zip(labels).foreach { case (feature, label) =>
      evaluation.eval(label, lstmNetwork.output(feature))
    }
    println(evaluation.stats())
    println(lstmNetwork.feedForward((new NDArray(Array(8f, 100000f, 8f, 11f), Array(1, 1, 4)))))
    println(lstmNetwork.output(new NDArray(Array(8f), Array(1, 1, 1))))
    println(lstmNetwork.output(new NDArray(Array(8f, 12f), Array(1, 1, 2))))
    println(lstmNetwork.output(new NDArray(Array(2f, 4f, 3f), Array(1, 1, 3))))
    //store
    ModelSerializer.writeModel(lstmNetwork, "src/main/resources/lstmnetwork", false)
  }

  //utility functions
  private def isProbability(data: Double): Boolean = data >= 0 && data <= 1

  private def prepareDataset(
      file: String,
      splitValidation: Double,
      splitTest: Double
  ): DataSetSplit = {
    require(isProbability(splitValidation) && isProbability(splitTest))

    val reader = new Iterator[List[Writable]] {
      val reader = new CSVRecordReader(',')
      reader.initialize(new FileSplit(new File(file)))
      override def hasNext: Boolean       = reader.hasNext
      override def next(): List[Writable] = reader.next().asScala.toList
    }.toList
    val shuffled = random.shuffle(reader).map(list => list.map(_.toFloat)).map { array =>
      val elements = array.reverse.tail.toArray
      val feature  = new NDArray(elements, Array(elements.length))
      val output   = new NDArray(Array(array.reverse.head))
      new DataSet(feature, output)
    }
    val (test, trainAndValidation) = shuffled.splitAt((shuffled.size * splitTest).toInt)
    val (validation, train)        = trainAndValidation.splitAt((trainAndValidation.size * splitValidation).toInt)
    DataSetSplit(
      wrapDataSetToSequenceIterator(train, batchSize),
      wrapDataSetToSequenceIterator(validation, batchSize),
      wrapDataSetToSequenceIterator(test, batchSize)
    )
  }

  private def attachUIServer(network: MultiLayerNetwork): Unit = {
    //gui configuration
    val uiServer     = UIServer.getInstance()
    val statsStorage = new InMemoryStatsStorage
    uiServer.attach(statsStorage)
    //add listener
    network.setListeners(new StatsListener(statsStorage))
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
