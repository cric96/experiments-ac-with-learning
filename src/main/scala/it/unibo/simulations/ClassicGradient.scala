package it.unibo.simulations

import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces.Node
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._

class ClassicGradient extends AggregateProgram with StandardSensors with ScafiAlchemistSupport with FieldUtils {

  type Hop       = Double
  type VisitNode = (Node[Any], Hop)
  private val minNumber    = 4
  private val feature      = 2
  private val totalData    = minNumber * feature
  private val defaultValue = 100000.0
  lazy val me: Node[Any]   = alchemistEnvironment.getNodeByID(mid())

  override def main(): Int = {

    rep(Double.PositiveInfinity) { data =>
      val neighborhood: Iterable[(Double, Double)] = excludingSelf
        .reifyField((nbr(data), nbrRange()))
        .values
      val sumOverNeighData = neighborhood.foldLeft((0.0, 0.0)) { (acc, data) =>
        (acc._1 + data._1, acc._2 + data._2)
      }
      val fillData             = List.fill(minNumber)((sumOverNeighData._1, sumOverNeighData._2))
      val minFirstNeighborhood = neighborhood.toList.sortBy(data => data._1 + data._2)
      val neighData            = (minFirstNeighborhood ++ fillData).take(minNumber)
      node.put("values", Seq(Double.PositiveInfinity))
      val result = classicGradient(target(me))
      if (!target(me)) {
        node.put("y", result)
        node.put("values", this.randomGen.shuffle(neighData).flatMap { case (left, right) => List(left, right) })
      }
      node.put("color", result)
      result
    }.toInt
  }

  def classicGradient(source: Boolean, metric: () => Double = nbrRange): Double =
    rep(Double.PositiveInfinity) { case d =>
      mux(source)(0.0)(minHoodPlus(nbr(d) + metric()))
    }

  private def target(node: Node[Any]): Boolean =
    new SimpleNodeManager[Any](node).get[Double]("target") == 1.0

}
