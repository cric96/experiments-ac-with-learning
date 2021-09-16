package it.unibo.casestudy

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist.ScafiAlchemistSupport
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.cpu.nativecpu.NDArray

import scala.concurrent.duration.FiniteDuration

class CrfNetworkEvaluator extends AggregateProgram with StandardSensors with ScafiAlchemistSupport with FieldUtils {
  private val cnnNetwork = ModelSerializer.restoreMultiLayerNetwork("src/main/resources/network")
  private val minNumber  = 4
  private val totalData  = minNumber

  override def main(): Any = {
    val reference = crf(isTarget)
    val networkResult = rep(-1.0) { data =>
      val neighborhood: Iterable[(Double, Double, Double)] =
        excludingSelf.reifyField((nbr(data), nbrRange(), nbrLag().toMillis / 1000.0)).values
      val sumOverNeighData = neighborhood.foldLeft((-1.0, -1.0, 0.0)) { (acc, data) =>
        (acc._1 + data._1, acc._2 + data._2, 0.0)
      }
      val fillData             = List.fill(totalData)((sumOverNeighData._1, sumOverNeighData._2, 0.0))
      val minFirstNeighborhood = neighborhood.toList.sortBy(data => data._1 + data._2)
      val neighData =
        (minFirstNeighborhood ++ fillData).take(totalData).flatMap { case (left, right, last) =>
          List(left, right, last)
        }
      node.put("cnn_in", neighData.mkString(","))
      val ndarray = new NDArray(neighData.map(_.toFloat).toArray, Array(1, 1, neighData.size))
      mux(isTarget)(0.0)(cnnNetwork.output(ndarray).getDouble(0L))

    }
    val cnnSquaredError = Math.pow(networkResult - reference, 2)
    node.put("reference", reference)
    node.put("result", networkResult)
    node.put("cnn_network", cnnSquaredError)
  }

  def crf(source: Boolean, raisingSpeed: Double = 5): Double = rep((Double.PositiveInfinity, 0.0)) { case (g, speed) =>
    mux(source)((0.0, 0.0)) {
      implicit def durationToDouble(fd: FiniteDuration): Double = fd.toMillis.toDouble / 1000.0
      case class Constraint(nbr: ID, gradient: Double, nbrDistance: Double)

      val constraints = foldhoodPlus[List[Constraint]](List.empty)(_ ++ _) {
        val (nbrg, d) = (nbr(g), nbrRange())
        mux(nbrg + d + speed * (nbrLag()) <= g)(List(Constraint(nbr(mid()), nbrg, d)))(List())
      }

      if (constraints.isEmpty) {
        (g + raisingSpeed * deltaTime(), raisingSpeed)
      } else {
        (constraints.map(c => c.gradient + c.nbrDistance).min, 0.0)
      }
    }
  }._1
  private def isTarget: Boolean = target == 1.0

  private def target: Double = sense[Double]("target")
}
