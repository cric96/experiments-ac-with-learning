package it.unibo.casestudy

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist.ScafiAlchemistSupport
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.cpu.nativecpu.NDArray

class NetworkEvaluator extends AggregateProgram with StandardSensors with ScafiAlchemistSupport with FieldUtils {
  private val cnnNetwork = ModelSerializer.restoreMultiLayerNetwork("src/main/resources/network")
  private val minNumber  = 4
  private val totalData  = minNumber

  override def main(): Any = {
    val reference = classicGradient(isTarget)
    val networkResult = rep(500.0) { data =>
      val neighborhood: Iterable[(Double, Double)] = excludingSelf.reifyField((nbr(data), nbrRange())).values
      val sumOverNeighData = neighborhood.foldLeft((500.0, 500.0)) { (acc, data) =>
        (acc._1 + data._1, acc._2 + data._2)
      }
      val fillData             = List.fill(totalData)((sumOverNeighData._1, sumOverNeighData._2))
      val minFirstNeighborhood = neighborhood.toList.sortBy(data => data._1 + data._2)
      val neighData =
        (minFirstNeighborhood ++ fillData).take(totalData).flatMap { case (left, right) => List(left, right) }
      node.put("cnn_in", neighData.mkString(","))
      val ndarray = new NDArray(neighData.map(_.toFloat).toArray, Array(1, 1, neighData.size))
      node.put(
        "verify",
        cnnNetwork.output(new NDArray(Array(400f, 16f, 150, 50f, 300f, 0f, 500f, 50f), Array(1, 1, 8)))
      )
      mux(isTarget)(0.0)(cnnNetwork.output(ndarray).getDouble(0L))

    }
    val cnnSquaredError = Math.pow(networkResult - reference, 2)
    node.put("reference", reference)
    node.put("result", networkResult)
    node.put("cnn_network", cnnSquaredError)
  }

  def classicGradient(source: Boolean, metric: () => Double = nbrRange): Double =
    rep(Double.PositiveInfinity) { case d =>
      mux(source)(0.0)(minHoodPlus(nbr(d) + metric()))
    }

  private def isTarget: Boolean = target == 1.0

  private def target: Double = sense[Double]("target")
}
