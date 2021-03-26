package it.unibo.alchemist.model.implementations.actions

import it.unibo.alchemist.model.implementations.actions.{
  AbstractAction,
  RunScafiProgram
}
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces.{
  Action,
  Context,
  Environment,
  Node,
  Position,
  Reaction
}

import java.io.{File, FileOutputStream, PrintWriter}
import scala.io.Source
import scala.jdk.CollectionConverters.IteratorHasAsScala

class ExtractCsv[T, P <: Position[P]](val env: Environment[T, P],
                                      val node: Node[T])
    extends AbstractAction[T](node) {
  override def cloneAction(node: Node[T], reaction: Reaction[T]): Action[T] =
    new ExtractCsv(env, node)

  override def execute(): Unit = {
    if (env.getSimulation.getTime.toDouble > 0) {
      val elements = env.getNodes.iterator().asScala.toList
      val csv = elements
        .map(new SimpleNodeManager[T](_))
        .map(
          node => s"${node.get("y")},${node.get("target")},${node.get("min")}"
        )
        .mkString("\n")
      val pw = new PrintWriter(
        new FileOutputStream(new File("output.csv"), true)
      )
      pw.append(csv + "\n")
      pw.close()
    }
  }
  override def getContext: Context = Context.GLOBAL
}
