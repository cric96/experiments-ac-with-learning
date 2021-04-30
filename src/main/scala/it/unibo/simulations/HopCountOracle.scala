package it.unibo.simulations

import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces.Node
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.common.StatisticsEvaluation

import scala.annotation.tailrec
import scala.collection.immutable.Queue
import scala.jdk.CollectionConverters._

class HopCountOracle
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport
    with FieldUtils
    with StatisticsEvaluation {

  type Hop       = Double
  type VisitNode = (Node[Any], Hop)
  private val delta      = 50
  lazy val me: Node[Any] = alchemistEnvironment.getNodeByID(mid())

  override def main(): Int = {
    rep(Double.PositiveInfinity) { data =>
      val statisticsIndex = extractStatisticsIndex(data)
      val (min, max, avg) = (statisticsIndex.min, statisticsIndex.max, statisticsIndex.avg)
      node.put("min", min)
      node.put("max", max)
      node.put("avg", avg)
      node.put("isTarget", booleanEncoding(target(me)))
      node.put("isNotTarget", booleanEncoding(!target(me)))
      node.put("status", sense[Double]("target"))
      val result = guess
      node.put("y", result)
      node.put("color", result * delta)
      result
    }.toInt

  }

  private def guess: Double = {
    alchemistEnvironment.getNeighborhood(me)
    breadthVisit(Queue((me, 0)), Set(me), node => target(node))
      .getOrElse[Double](Double.PositiveInfinity)
  }

  @tailrec
  private def breadthVisit(
      frontier: Queue[VisitNode],
      visited: Set[Node[Any]] = Set.empty,
      findCondition: Node[Any] => Boolean
  ): Option[Hop] = {
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
        val visitedUpdated   = visited ++ neighbourToVisit
        val queueUpdated: Queue[VisitNode] = frontier.tail ++ neighbourToVisit
          .map(node => (node, next))
        breadthVisit(queueUpdated, visitedUpdated, findCondition)
    }
  }

  private def target(node: Node[Any]): Boolean =
    new SimpleNodeManager[Any](node).get[Double]("target") == 1.0
}
