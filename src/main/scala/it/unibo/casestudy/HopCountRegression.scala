package it.unibo.casestudy

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import smile.read
import smile.regression._

import java.nio.file.Path

class HopCountRegression
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport {
  private val model: LinearModel =
    read
      .xstream(Path.of(getClass.getResource("/model").toURI))
      .asInstanceOf[LinearModel]
  private val delta = 50
  override def main = {
    rep(Double.PositiveInfinity) { data =>
      {
        val minData = minHood(nbr(data))
        val result = model.predict(Array(target, minData))
        node.put("color", result * delta)
        mux(isTarget) { 0.0 } { result } //not right
        //right way : result
      }
    }.toInt
  }

  private def isTarget: Boolean = target == 1.0

  private def target: Double = sense[Double]("target")
}
