package it.unibo.simulations
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces.Node
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._

import scala.annotation.tailrec
import scala.collection.immutable.Queue
import scala.jdk.CollectionConverters._

class HopCountOracle
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport {

  type Hop = Double
  type VisitNode = (Node[Any], Hop)
  lazy val me: Node[Any] = alchemistEnvironment.getNodeByID(mid())
  override def main(): Int = {
    rep(Double.PositiveInfinity) { data =>
      val min = minHood(nbr(data))
      node.put("min", min)
      node.put("status", sense[Double]("target"))
      val result = guess(min)
      node.put("y", result)
      node.put("color", result * 50)

      result
    }.toInt

  }

  private def guess(data: Double): Double = {
    alchemistEnvironment.getNeighborhood(me)
    breadthVisit(Queue((me, 0)), Set(me), node => target(node))
      .getOrElse[Double](Double.PositiveInfinity)
  }

  @tailrec
  private def breadthVisit(frontier: Queue[VisitNode],
                           visited: Set[Node[Any]] = Set.empty,
                           findCondition: Node[Any] => Boolean): Option[Hop] = {
    frontier.headOption match {
      case None                                     => None
      case Some((node, hop)) if findCondition(node) => Some(hop)
      case Some((node, hop)) =>
        val next = hop + 1
        val neighbour = alchemistEnvironment
          .getNeighborhood(node)
          .getNeighbors
          .iterator()
          .asScala
          .toList
        val neighbourToVisit = neighbour.filterNot(visited.contains)
        val visitedUpdated = visited ++ neighbourToVisit
        val queueUpdated: Queue[VisitNode] = frontier.tail ++ neighbourToVisit
          .map(node => (node, next))
        breadthVisit(queueUpdated, visitedUpdated, findCondition)
    }
  }

  private def target(node: Node[Any]): Boolean =
    new SimpleNodeManager[Any](node).get[Double]("target") == 1.0
}
