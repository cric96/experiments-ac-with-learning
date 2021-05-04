package it.unibo.learning

import it.unibo.learning.DeepNetworks.DataSetSplit
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.api.BaseTrainingListener
import org.nd4j.evaluation.regression.RegressionEvaluation

import scala.jdk.CollectionConverters.IteratorHasAsScala

class ValidationScoreListener(dataset: DataSetSplit) extends BaseTrainingListener {
  private val testElements = dataset.validationSet.asScala.toList

  override def onEpochEnd(model: Model): Unit = {
    val mlp        = model.asInstanceOf[MultiLayerNetwork]
    val evaluation = new RegressionEvaluation()
    val labels     = testElements.map(_.getLabels)
    val feature    = testElements.map(_.getFeatures)
    feature.zip(labels).foreach { case (feature, label) =>
      evaluation.eval(label, mlp.output(feature))
    }
    println(evaluation.stats())
  }
}
