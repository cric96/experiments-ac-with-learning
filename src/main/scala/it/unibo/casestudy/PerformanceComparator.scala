package it.unibo.casestudy

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist.ScafiAlchemistSupport
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.learning.Dataset
import it.unibo.learning.Dataset._
import org.deeplearning4j.util.ModelSerializer
import smile.data.Tuple
import smile.read
import smile.regression.DataFrameRegression

import java.nio.file.Path

class PerformanceComparator extends AggregateProgram with StandardSensors with ScafiAlchemistSupport {

  private val linear        = loadRegression("linear")
  private val lasso         = loadRegression("lasso")
  private val randomForest  = loadRegression("random_forest")
  private val ridge         = loadRegression("ridge")
  private val gradientBoost = loadRegression("gradient_boost")
  private val network       = ModelSerializer.restoreMultiLayerNetwork("network")

  override def main(): Any = {
    val reference = hopCountLike(tuple => tuple.min + 1)
    List(
      (linear, "linear"),
      (lasso, "lasso"),
      (randomForest, "randomForest"),
      (ridge, "ridge"),
      (gradientBoost, "gradientBoost")
    ).foreach { case (regression, name) =>
      evalAndStoreError(reference, regression, name)
    }
    //network evaluation
    val deepNNResult = hopCountLike(tuple => network.output(tuple.asNDArray).getDouble(0L))
    val squaredError = Math.pow((deepNNResult - reference), 2)
    node.put("network", squaredError)
  }

  private def evalAndStoreError(reference: Double, regression: DataFrameRegression, name: String): Unit = {
    val regressionResult = hopCountLike(usingRegression(regression, _))
    val squaredError     = Math.pow((regressionResult - reference), 2)
    node.put(name, squaredError)
  }

  private def hopCountLike(eval: Tuple => Double): Double =
    rep(Double.PositiveInfinity) { data =>
      val minData = minHood(nbr(data))
      val tuple   = Dataset.createTuple(target, minData)
      val result  = Math.round(eval(tuple)).toInt
      mux(isTarget)(0)(result) //not right
    //right way : result
    }

  private def usingRegression(regression: DataFrameRegression, data: Tuple): Int =
    Math.round(regression.predict(data)).toInt

  private def loadRegression(name: String): DataFrameRegression = {
    read
      .xstream(Path.of(getClass.getResource(s"/$name").toURI))
      .asInstanceOf[DataFrameRegression]
  }

  private def isTarget: Boolean = target == 1.0

  private def target: Double = sense[Double]("target")
}
