package it.unibo.casestudy

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.learning.Dataset
import smile.read
import smile.regression._

import java.nio.file.Path

class HopCountRegression extends AggregateProgram with StandardSensors with ScafiAlchemistSupport {
  private val modelName: String = "linear" //linear, gradient, lasso, random_forest..
  private val model: DataFrameRegression =
    read
      .xstream(Path.of(getClass.getResource(s"/$modelName").toURI))
      .asInstanceOf[DataFrameRegression]
  private val delta = 50

  override def main(): Double = {
    rep(Double.PositiveInfinity) { data =>
      val minData = minHood(nbr(data))
      val tuple   = Dataset.createTuple(target, minData)
      val result  = Math.round(model.predict(tuple)).toInt
      node.put("color", result * delta)
      mux(isTarget)(0)(result) //not right
    //right way : result
    }
  }

  private def isTarget: Boolean = target == 1.0

  private def target: Double = sense[Double]("target")
}
