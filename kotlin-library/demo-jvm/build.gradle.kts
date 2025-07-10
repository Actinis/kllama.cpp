plugins {
    kotlin("jvm")
    application
}

group = "io.actinis.kllama_cpp.demo"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation(project(":library"))

    implementation(libs.clikt)
}

application {
    mainClass.set("io.actinis.kllama_cpp.demo.jvm.MainKt")

    val osName = System.getProperty("os.name")
    val nativeLibPath = when {
        osName.startsWith("Linux") -> "../native/build/linux/x86_64/release/library"
        osName.startsWith("Windows") -> "../native/build/windows/x86_64/release/library"
        osName.startsWith("Mac") -> "../native/build/osx/aarch64/release/library"
        else -> throw GradleException("Unsupported OS for native demo: $osName")
    }
    applicationDefaultJvmArgs = listOf("-Djava.library.path=$nativeLibPath")
}

tasks.test {
    useJUnitPlatform()
}

kotlin {
    jvmToolchain(17)
}