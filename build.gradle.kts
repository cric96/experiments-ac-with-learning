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
}

tasks.withType<ScalaCompile> {
    sourceCompatibility = "1.11"
    targetCompatibility = "1.11"
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
makeTest(name="simulation", file = "simulation", time = 100.0, vars = setOf("random"), taskSize = 2800)
makeTest(name="helloGUI", file = "simulation")

defaultTasks("fatJar")
