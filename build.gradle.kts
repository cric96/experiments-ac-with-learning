import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar
import java.io.ByteArrayOutputStream
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

plugins {
    java
    scala
    id("com.github.johnrengelman.shadow") version "4.0.3"
    idea
    kotlin("jvm") version "1.3.50"
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("it.unibo.scafi:scafi-core_2.13:0.3.3")
    //alchemist and scafi dependency
    implementation("it.unibo.alchemist:alchemist:10.0.1")
    implementation("it.unibo.alchemist:alchemist-swingui:10.0.1") //without this dependency, the simulation doesn't produce data
    implementation("it.unibo.alchemist:alchemist-incarnation-scafi:10.0.1")
    implementation("org.scala-lang:scala-library:2.13.2")
    //ml dependency
    implementation("com.github.haifengl:smile-scala_2.13:2.6.0")
    //native dependency for smile
    implementation("org.bytedeco:javacpp:1.5.4:linux-x86_64")
    implementation("org.bytedeco:javacpp:1.5.4:windows-x86_64")
    implementation("org.bytedeco:javacpp:1.5.4:macosx-x86_64")
    //openblas
    implementation("org.bytedeco:openblas:0.3.10-1.5.4:linux-x86_64")
    implementation("org.bytedeco:openblas:0.3.10-1.5.4:windows-x86_64")
    implementation("org.bytedeco:openblas:0.3.10-1.5.4:macosx-x86_64")
    //arpack
    implementation("org.bytedeco:arpack-ng:3.7.0-1.5.4:linux-x86_64")
    implementation("org.bytedeco:arpack-ng:3.7.0-1.5.4:windows-x86_64")
    implementation("org.bytedeco:arpack-ng:3.7.0-1.5.4:macosx-x86_64")
    //deep learning libraries
    implementation("org.deeplearning4j:deeplearning4j-core:1.0.0-beta7")
    implementation("org.nd4j:nd4j-native-platform:1.0.0-beta7")
    implementation("org.deeplearning4j:deeplearning4j-ui:1.0.0-beta7")
}

tasks.withType<ScalaCompile> {
    sourceCompatibility = "1.11"
    targetCompatibility = "1.11"
    //sane scalac configuration from http://tpolecat.github.io/2017/04/25/scalac-flags.html
    scalaCompileOptions.additionalParameters = listOf(
        "-deprecation",                      // Emit warning and location for usages of deprecated APIs.
        "-encoding", "utf-8",                // Specify character encoding used by source files.
        "-explaintypes",                     // Explain type errors in more detail.
        "-feature",                          // Emit warning and location for usages of features that should be imported explicitly.
        "-language:existentials",            // Existential types (besides wildcard types) can be written and inferred
        "-language:experimental.macros",     // Allow macro definition (besides implementation and application)
        "-language:higherKinds",             // Allow higher-kinded types
        "-language:implicitConversions",     // Allow definition of implicit functions called views
        "-unchecked",                        // Enable additional warnings where generated code depends on assumptions.
        "-Xcheckinit",                       // Wrap field accessors to throw an exception on uninitialized access.
        "-Xfatal-warnings",                  // Fail the compilation if there are any warnings.
        "-Xlint:adapted-args",               // Warn if an argument list is modified to match the receiver.
        "-Xlint:constant",                   // Evaluation of a constant arithmetic expression results in an error.
        "-Xlint:delayedinit-select",         // Selecting member of DelayedInit.
        "-Xlint:doc-detached",               // A Scaladoc comment appears to be detached from its element.
        "-Xlint:inaccessible",               // Warn about inaccessible types in method signatures.
        "-Xlint:infer-any",                  // Warn when a type argument is inferred to be `Any`.
        "-Xlint:missing-interpolator",       // A string literal appears to be missing an interpolator id.
        "-Xlint:nullary-unit",               // Warn when nullary methods return Unit.
        "-Xlint:option-implicit",            // Option.apply used implicit view.
        "-Xlint:package-object-classes",     // Class or object defined in package object.
        "-Xlint:poly-implicit-overload",     // Parameterized overloaded implicit methods are not visible as view bounds.
        "-Xlint:private-shadow",             // A private field (or class parameter) shadows a superclass field.
        "-Xlint:stars-align",                // Pattern sequence wildcard must align with sequence component.
        "-Xlint:type-parameter-shadow",      // A local type parameter shadows a type already in scope.
        "-Ywarn-dead-code",                  // Warn when dead code is identified.
        "-Ywarn-extra-implicit",             // Warn when more than one implicit parameter section is defined.
        "-Ywarn-unused:implicits",           // Warn if an implicit parameter is unused.
        "-Ywarn-unused:imports",             // Warn if an import selector is not referenced.
        "-Ywarn-unused:locals",              // Warn if a local definition is unused.
        "-Ywarn-unused:params",              // Warn if a value parameter is unused.
        "-Ywarn-unused:patvars",             // Warn if a variable bound in a pattern is unused.
        "-Ywarn-value-discard"               // Warn when non-Unit expression results are unused.
    )
}

idea {
    module {
        isDownloadJavadoc = true
        isDownloadSources = true
    }
}

sourceSets.getByName("main") {
    resources {
        srcDirs("src/main/protelis")
    }
}

// Needed to avoid error:
// > shadow.org.apache.tools.zip.Zip64RequiredException: archive contains more than 65535 entries.
tasks.withType<ShadowJar> {
    isZip64 = true
    classifier = null
    version = null
    // baseName = "anotherBaseName"
}

tasks.register<Jar>("fatJar") {
    manifest {
        attributes(mapOf(
                "Implementation-Title" to "Alchemist",
                "Implementation-Version" to rootProject.version,
                "Main-Class" to "it.unibo.alchemist.Alchemist",
                "Automatic-Module-Name" to "it.unibo.alchemist"
        ))
    }
    archiveBaseName.set("${rootProject.name}-redist")
    isZip64 = true
    from(configurations.runtimeClasspath.get().map { if (it.isDirectory) it else zipTree(it) }) {
        // remove all signature files
        exclude("META-INF/")
        exclude("ant_tasks/")
        exclude("about_files/")
        exclude("help/about/")
        exclude("build")
        exclude("out")
        exclude("bin")
        exclude(".gradle")
        exclude("build.gradle.kts")
        exclude("gradle")
        exclude("gradlew")
        exclude("gradlew.bat")
    }
    with(tasks.jar.get() as CopySpec)
}

fun makeTest(
        file: String,
        name: String = file,
        sampling: Double = 1.0,
        effect : String = "",
        time: Double = Double.POSITIVE_INFINITY,
        vars: Set<String> = setOf(),
        maxHeap: Long? = null,
        taskSize: Int = 1024,
        threads: Int? = null,
        debug: Boolean = false
) {
    val heap: Long = maxHeap ?: if (System.getProperty("os.name").toLowerCase().contains("linux")) {
        ByteArrayOutputStream()
                .use { output ->
                    exec {
                        executable = "bash"
                        args = listOf("-c", "cat /proc/meminfo | grep MemAvailable | grep -o '[0-9]*'")
                        standardOutput = output
                    }
                    output.toString().trim().toLong() / 1024
                }
                .also { println("Detected ${it}MB RAM available.") }  * 9 / 10
    } else {
        // Guess 10GB RAM of which 2 used by the OS
        10 * 1024L
    }

    val threadCount = threads ?: maxOf(1, minOf(Runtime.getRuntime().availableProcessors(), heap.toInt() / taskSize ))
    println("Running on $threadCount threads")

    val today = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"))

    task<JavaExec>("$name") {
        classpath = sourceSets["main"].runtimeClasspath
        classpath("src/main/protelis")
        main = "it.unibo.alchemist.Alchemist"
        maxHeapSize = "${heap}m"
        jvmArgs("-XX:+AggressiveHeap")
        jvmArgs("-XX:-UseGCOverheadLimit")
        //jvmArgs("-XX:+UnlockExperimentalVMOptions", "-XX:+UseCGroupMemoryLimitForHeap") // https://stackoverflow.com/questions/38967991/why-are-my-gradle-builds-dying-with-exit-code-137
        if (debug) {
            jvmArgs("-agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=1044")
        }
        File("data").mkdirs()
        args(
                "-y", "src/main/yaml/${file}.yml",
                "-t", "$time",
                "-e", "data/${today}-${name}",
                "-p", threadCount,
                "-i", "$sampling"
        )

        if (vars.isNotEmpty()) {
            args("-b", "-var", *vars.toTypedArray(), "--headless")
        }

        if(effect != "") {
            args("-g", "./src/main/resources/${effect}.aes")
        }
    }
    /*tasks {
        "runTests" {
            dependsOn("$name")
        }
    }*/
}

makeTest(name="simulationGUI", file = "simulation", effect = "effect")
makeTest(name="simulation", file = "simulation", time = 11.0, vars = setOf("random"), threads = 1)
makeTest(name="multiValidation", file = "multi_validation", time = 100.0, vars = setOf("random"))
makeTest(name="multiValidationGUI", file = "multi_validation", effect = "effect")
defaultTasks("fatJar")
