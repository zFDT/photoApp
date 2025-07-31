## 添加一些备忘录

### 如何获取 ensorflow-lite 的C++

1.

教程指向的 [链接](https://central.sonatype.com/artifact/org.tensorflow/tensorflow-lite-support/overview)
不可靠

2. 通过AS的方式来获取链接
    1. [libs.versions.toml](gradle/libs.versions.toml) 配置
```xml
   tensorflow = "2.16.0 "
   tensorflow-lite = "2.16.0"
   tensorflow-lite = { group = "org.tensorflow", name = "tensorflow-lite", version.ref = "
   tensorflow-lite" }
   tensorflow-lite-support = { group = "org.tensorflow", name = "tensorflow-lite-support", version = "
   0.4.4" }
   tensorflow-lite-metadata = { group = "org.tensorflow", name = "tensorflow-lite-metadata",
   version = "0.4.4" }
```
   2. [build.gradle](app/build.gradle) 配置
```xml
    implementation(libs.tensorflow.lite)
    implementation(libs.tensorflow.lite.support)
    implementation(libs.tensorflow.lite.metadata)
```
   3. 通过 [tensorflow-lite](https://repo.maven.apache.org/maven2/org/tensorflow/tensorflow-lite/2.16.1/tensorflow-lite-2.16.1.aar)
   下载 arr 包,来解压到项目使用