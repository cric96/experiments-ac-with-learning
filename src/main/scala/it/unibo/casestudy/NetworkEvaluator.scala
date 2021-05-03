package it.unibo.casestudy

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist.ScafiAlchemistSupport
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.cpu.nativecpu.NDArray

class NetworkEvaluator extends AggregateProgram with StandardSensors with ScafiAlchemistSupport with FieldUtils {

  private val network = ModelSerializer.restoreMultiLayerNetwork("src/main/resources/network")

  override def main(): Any = {
    val reference = hopCountLike(_.min + 1)
    val networkResult = hopCountLike { elements =>
      val ndarray = new NDArray(elements.map(_.toFloat).toArray, Array(1, 1, elements.size))
      network.output(ndarray).getDouble(0L)
    }
    val squaredError = Math.pow(networkResult - reference, 2)
    node.put("network", squaredError)
  }

  private def hopCountLike(expr: Iterable[Double] => Double): Double =
    rep(Double.PositiveInfinity) { data =>
      val values = excludingSelf.reifyField(data).values
      val result = if (values.nonEmpty) expr(values).toInt else data.toInt
      mux(isTarget)(0)(result) //not right
    //right way : result
    }

  private def isTarget: Boolean = target == 1.0

  private def target: Double = sense[Double]("target")
}
