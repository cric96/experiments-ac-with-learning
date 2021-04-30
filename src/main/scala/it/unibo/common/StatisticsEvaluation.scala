package it.unibo.common

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._

trait StatisticsEvaluation {
  self: AggregateProgram with FieldUtils =>

  case class StatisticsIndex(min: Double, max: Double, avg: Double)

  def extractStatisticsIndex[A: Numeric](expression: A): StatisticsIndex = {
    val data = excludingSelf.reifyField(nbr(expression)).values
    val (min, max, avg) = (
      data.minOption.map(Numeric[A].toDouble).getOrElse(Double.PositiveInfinity),
      data.maxOption.map(Numeric[A].toDouble).getOrElse(Double.PositiveInfinity),
      data
        .reduceOption((a, b) => Numeric[A].plus(a, b))
        .map(result => Numeric[A].toDouble(result))
        .map(_ / data.size)
        .getOrElse(Double.PositiveInfinity)
    )
    StatisticsIndex(min, max, avg)
  }

  def booleanEncoding(boolean: Boolean): Double =
    if (boolean) {
      1.0
    } else {
      0.0
    }
}
