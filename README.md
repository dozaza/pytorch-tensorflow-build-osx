A tutorial of how to build Pytorch & Tensorflow with GPU support locally on OSX 10.13

---

Since Pytorch & Tensorflow no longer supported on OSX, I tried to build it on my own machine.

Python version: 3.6.5

# For Tensorflow 1.10
1. Install CUDA >= 9.2, cudNN >= 7.1, bazel(latest), Xcode == 8.3.3
   - XCode should be 8.3.3 or build would success, but you would get segument fault when using it.
     use ```sudo xcode-select -s /Applications/Xcode8.app``` to change your XCode. (Suppose both Xcode8 & Xcode9 in the environment)
   - cudNN installation:
   ```
   sudo cp cuda/include/cudnn.h /usr/local/cuda/include
   sudo cp cuda/lib/libcudnn* /usr/local/cuda/lib
   sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib/libcudnn*
   ```
   - NCCL installation
   ```
   sudo mkdir -p /usr/local/nccl
   sudo mv nccl_2.2.13-1+cuda9.2_x86_64/* /usr/local/nccl
   sudo mkdir -p /usr/local/include/third_party/nccl
   sudo ln -s /usr/local/nccl/include/nccl.h /usr/local/include/third_party/nccl
   ```
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
9. Start building: 
    
    ```bazel build --config=cuda --config=opt --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH //tensorflow/tools/pip_package:build_pip_package```
10. Generate wheel file: 
    
    ```bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tensorflow_pkg```
11. Install package: 
    
    ```pip install ./tensorflow_pkg/tensorflow-1.10.1-cp36-cp36m-macosx_10_13_x86_64.whl```
    
    Wheel file name depends on your environment. 
12. When finished, get out of tensorflow folder and run
    ```
    python
    >>> import tensorflow as tf
    >>> tf.Session() 
    ```
    GPU info will be displayed in python repl.
    
# For Pytorch 0.4.1
1. Install CUDA >= 9.2, cudNN >= 7.1, NCCL >= 2.2.13, bazel(latest), Xcode == 9.2
   
   Same operation as Step 1 in Tensorflow tutorial above.
2. Install python dependencies 
    
    ```pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing```
3. Check System Integrity Protection status is 'disabled' by running ```csrutil status```, if not execute ```csrutil disable``` in recovery mode.
4. Install anaconda3.
5. Make sure followings is in your bashrc or zshrc:
    ```
    export CUDA_HOME=/usr/local/cuda
    export DYLD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/nvvm/lib:$CUDA_HOME/extras/CUPTI/lib:/usr/local/nccl/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}
    export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$LD_LIBRARY_PATH
    export PATH=$PATH::$CUDA_HOME/bin
    ```
    ```
    export CMAKE_PREFIX_PATH=your_anaconda_root_address # ~/anaconda3
    ```
6. Run ```MACOSX_DEPLOYMENT_TARGET=10.13 CC=clang CXX=clang++ python setup.py install``` in pytorch folder.
7. When finished, get out of pytorch folder and run
   ```
   python
   >>> import torch
   >>> torch.cuda.is_available()
   ```
   True will be displayed in python repl.
