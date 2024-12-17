# conda create -n ptcept python=3.8 -y
# conda activate ptcept
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-scatter -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu118
# pip install spconv-cu120

# PTv1 & PTv2 or precise eval
cd libs/pointops
# # usual
# python setup.py install
# # docker & multi GPU arch
# TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9" python setup.py install
cd ../..

# Open3D (visualization, optional)
pip install open3d

# Warp
pip install warp-lang

# MMCV
pip install -U openmim
mim install mmcv==2.0.0rc4
mim install mmdet==3.0.0
mim install mmdet3d==1.4.0

conda install -c bioconda google-sparsehash 

# # swin 3d
# cd libs
# git clone https://github.com/microsoft/Swin3D.git
# cd Swin3D
# pip install -r requirements.txt
# python setup.py install
# cd ../..

# # MinkowskiEngine
# cd libs/minkowski
# apt install libopenblas-dev
# conda install -c conda-forge openblas
# MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="8.9" python setup.py install --blas=openblas

# Flash Attention

# install by setup.py
# cd ../../
# cd libs/flash-attention
# MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="8.9" python setup.py install

# else install by pip
# flash attention for cu123 torch2.1
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
# pip install flash_attn-2.6.3+cu123torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

# flash attention for cu118 torch2.0
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install flash_attn-2.6.3+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl