### Running demo

Build the native part, and then:

```shell
java -Djava.library.path=/src/kotlin/files-renamer/kllama-cpp/native/build/linux/x86_64/release/library io.actinis.kllama_cpp.demo.jvm.MainKt --model /home/user/Downloads/gemma-3-4b-it-q4_0.gguf --mmproj /home/user/Downloads/mmproj-model-f16-4B.gguf --image /home/user/Screenshots/Screenshot_20250710_142150.png --prompt What do you see on this image? Please describe all the details --temperature 0
```