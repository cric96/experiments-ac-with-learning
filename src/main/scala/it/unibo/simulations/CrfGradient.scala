package it.unibo.simulations

import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces.Node
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._

import scala.concurrent.duration.FiniteDuration

class CrfGradient extends AggregateProgram with StandardSensors with ScafiAlchemistSupport with FieldUtils {

  type Hop       = Double
  type VisitNode = (Node[Any], Hop)
  private val minNumber  = 4
  private val feature    = 3
  private val totalData  = minNumber * feature
  lazy val me: Node[Any] = alchemistEnvironment.getNodeByID(mid())

  override def main(): Int = {

    rep(Double.PositiveInfinity) { data =>
      val neighborhood: Iterable[(Double, Double, Double)] = excludingSelf
        .reifyField((nbr(data), nbrRange(), nbrLag().toMillis.toDouble / 1000.0))
        .values
      val sumOverNeighData = neighborhood.foldLeft((0.0, 0.0, 0.0)) { (acc, data) =>
        (acc._1 + data._1, acc._2 + data._2, acc._3)
      }
      val fillData             = List.fill(minNumber)((sumOverNeighData._1, sumOverNeighData._2, 0.0))
      val minFirstNeighborhood = neighborhood.toList.sortBy(data => data._1 + data._2)
      val neighData            = (minFirstNeighborhood ++ fillData).take(minNumber)
      node.put("values", Seq(Double.PositiveInfinity))
      val result = crf(target(me))
      if (!target(me)) {
        node.put("y", result)
        node.put(
          "values",
          this.randomGen.shuffle(neighData).flatMap { case (left, right, last) => List(left, right, last) }
        )
      }
      node.put("color", result)
      result
    }.toInt
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

  private def target(node: Node[Any]): Boolean =
    new SimpleNodeManager[Any](node).get[Double]("target") == 1.0

}
