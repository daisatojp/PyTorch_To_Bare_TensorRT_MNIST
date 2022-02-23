# PyTorch_To_Bare_TensorRT_MNIST

An MNIST example for conversion from PyTorch to a network built by TensorRT C++ API.

# Run

I'm working on the NVIDIA's docker container for TensorRT [here](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_21-09.html). For more detail, please see the [Dockerfile](https://github.com/daisatojp/PyTorch_To_Bare_TensorRT_MNIST/blob/main/.devcontainer/Dockerfile).

```bash
git clone https://github.com/daisatojp/PyTorch_To_Bare_TensorRT_MNIST.git
cd PyTorch_To_Bare_TensorRT_MNIST

python train.py

wget https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.10.2%2Bcu113.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.10.2+cu113.zip
rm libtorch-cxx11-abi-shared-with-deps-1.10.2+cu113.zip
mv libtorch libtorch_v1.10.2

mkdir build && cd build
cmake -G "Ninja" ..
ninja -v
./main
# It should print like this
# 
# Input:
# 
#                             
#                             
#                             
#                             
#                             
#                 =+*.*@@=    
#         ..-+*@@@@@%*@@#:    
#        .@@@@@@@@@@---:.     
#         %@@@@@##@@          
#         -*=@@#  .+          
#            +@-              
#            +@#              
#             #@:             
#             .@%*=           
#              -@@@=          
#               .#@@+.        
#                 -@@#        
#                  @@@:       
#               .+#@@#        
#             .+%@@@@#        
#            =%@@@@#-         
#          :%@@@@#-           
#        *%@@@@#-             
#     :*%@@@@@+               
#     +@@@%++                 
#                             
#                             
#                             
# 
# Output:
# 
# 0: 
# 1: 
# 2: 
# 3: 
# 4: 
# 5: **********
# 6: 
# 7: 
# 8: 
# 9: 

# To check C++ Tensor in Python
cd ..
python check_cpp_tensor.py
```
