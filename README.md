A tutorial of how to build Pytorch & Tensorflow with GPU support locally on OSX 10.13

---

Since Pytorch & Tensorflow no longer supported on OSX, I tried to build it on my own machine.

# For Tensorflow 1.10
1. Install CUDA >= 9.2, cudNN >= 7.1, bazel(latest), Xcode == 8.3.3
   Attention: XCode should be 8.3.3 or build would success, but you would get segument fault when using it.
   use ```sudo xcode-select -s /Applications/Xcode8.app``` to change your XCode. (Suppose both Xcode8 & Xcode9 in the environment)

2. Install python dependencies ```pip install six numpy wheel```
3. Install GNU support tools ```brew install coreutils```
4. Check System Integrity Protection status is 'disabled' by running ```csrutil status```, if not execute ```csrutil disable``` in recovery mode.
5. Remove all **__align(sizeof(T))__** from following files:
   - tensorflow/core/kernels/depthwise_conv_op_gpu.cu.cc
   - tensorflow/core/kernels/split_lib_gpu.cu.cc
   - tensorflow/core/kernels/concat_lib_gpu.impl.cu.cc
   For example, extern shared __align(sizeof(T))__ unsigned char smem[]; => extern shared unsigned char smem[];
6. Run ```./configure```, make sure:
   **CUDA SDK version** is your CUDA version(maybe 9.2), **build TensorFlow with CUDA support** is y, **cuDNN version** is your cuDnn version(maybe 7.1), **Cuda compute capabilities** depends on your GPU, check it out at [click me](https://developer.nvidia.com/cuda-gpus)
7. Make sure followings is in your bashrc or zshrc:
   ```
   export CUDA_HOME=/usr/local/cuda
   export DYLD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/nvvm/lib:$CUDA_HOME/extras/CUPTI/lib:/usr/local/nccl/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}
   export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$LD_LIBRARY_PATH
   export PATH=$PATH::$CUDA_HOME/bin
   ```
8. Comment out ```linkopts = ["-lgomp"]``` in file ```third_party/gpus/cuda/BUILD.tpl```.
9. Start building: ```bazel build --config=cuda --config=opt --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH //tensorflow/tools/pip_package:build_pip_package```
10. Generate wheel file: ```bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tensorflow_pkg```
11. Install package: ```pip install ./tensorflow_pkg/tensorflow-1.10.1-cp36-cp36m-macosx_10_13_x86_64.whl```. Wheel file name depends on your environment. 
