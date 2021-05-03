package it.unibo.casestudy

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist.ScafiAlchemistSupport
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import org.deeplearning4j.util.ModelSerializer

class NetworkEvaluator extends AggregateProgram with StandardSensors with ScafiAlchemistSupport with FieldUtils {

  private val network = ModelSerializer.restoreMultiLayerNetwork("network")

  override def main(): Any = {
    println(network) //to remove
    val reference     = hopCountLike(_.min + 1)
    val networkResult = hopCountLike(_ => 0) //todo
    val squaredError  = Math.pow((networkResult - reference), 2)
    node.put("network", squaredError)
  }

  private def hopCountLike(expr: Iterable[Double] => Double): Double =
    rep(Double.PositiveInfinity) { data =>
      val values = excludingSelf.reifyField(data).values
      val result = expr(values).toInt
      mux(isTarget)(0)(result) //not right
    //right way : result
    }

  private def isTarget: Boolean = target == 1.0

  private def target: Double = sense[Double]("target")
}
