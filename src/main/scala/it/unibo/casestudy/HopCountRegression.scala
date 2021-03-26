package it.unibo.casestudy

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import smile.read
import smile.regression._

import java.nio.file.Path

class HopCountRegression extends AggregateProgram with StandardSensors {
  private val model: LinearModel =
    read
      .xstream(Path.of(getClass.getResource("/model").toURI))
      .asInstanceOf[LinearModel]
  override type MainResult = Int

  override def main = {
    rep(Double.PositiveInfinity) { data =>
      {
        val minData = minHood(nbr(data))
        model.predict(Array(minData, isTarget))
      }
    }.toInt
  }

  private def isTarget: Double = sense[Double]("target")
}
