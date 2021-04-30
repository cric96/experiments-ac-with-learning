package it.unibo.alchemist.model.implementations.actions

import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces._

import java.io.File
import java.io.FileOutputStream
import java.io.PrintWriter
import scala.jdk.CollectionConverters.IteratorHasAsScala

class ExtractCsv[T, P <: Position[P]](val env: Environment[T, P], val node: Node[T], labels: List[String])
    extends AbstractAction[T](node) {

  override def cloneAction(node: Node[T], reaction: Reaction[T]): Action[T] =
    new ExtractCsv(env, node, labels)

  override def execute(): Unit = {
    if (env.getSimulation.getTime.toDouble > 0) {
      val elements = env.getNodes.iterator().asScala.toList
      val csv = elements
        .map(new SimpleNodeManager[T](_))
        .map { node =>
          adjustValues(labels, node).map(_.toString).mkString(",")
        }
        .mkString("\n")
      val pw = new PrintWriter(
        new FileOutputStream(new File("output.csv"), true)
      )
      pw.append(csv + "\n")
      pw.close()
    }
  }
  override def getContext: Context = Context.GLOBAL

  private def adjustValues(label: List[String], node: SimpleNodeManager[T]): List[Double] =
    label.map(node.get[Double]).collect {
      case value if value.isInfinity => Int.MaxValue.toDouble
      case value                     => value
    }
}
