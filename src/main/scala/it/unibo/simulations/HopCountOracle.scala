package it.unibo.simulations

import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces.Node
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._

class HopCountOracle extends AggregateProgram with StandardSensors with ScafiAlchemistSupport with FieldUtils {

  type Hop       = Double
  type VisitNode = (Node[Any], Hop)
  private val minNumber    = 5
  private val totalData    = minNumber
  private val defaultValue = -1.0
  private val fillData     = List.fill(totalData)((defaultValue, defaultValue))
  lazy val me: Node[Any]   = alchemistEnvironment.getNodeByID(mid())

  override def main(): Int = {

    rep(Double.PositiveInfinity) { data =>
      val neighborhood: Iterable[(Double, Double)] = excludingSelf.reifyField(nbr((data, nbrRange()))).values
      val minFirstNeighborhood                     = neighborhood.toList.sortBy(data => data._1 + data._2)
      val neighData                                = (minFirstNeighborhood ++ fillData).take(totalData)
      val shuffleData                              = this.randomGen.shuffle(neighData)
      val flatten                                  = shuffleData.flatMap(data => List(data._1, data._2))
      val values                                   = this.randomGen.shuffle(flatten) //fixed size
      node.put("values", values)
      val result = classicGradient(target(me))
      node.put("y", result)
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
