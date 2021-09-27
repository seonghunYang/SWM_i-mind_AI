FROM nvidia/cuda:11.0.3-devel-ubuntu18.04

ENV PATH /opt/conda/bin:$PATH

# Install dependencies of miniconda
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 curl git gcc g++ libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
    # echo "source activate CatchNet" >> ~/.bashrc

# Conda environment setup
# RUN conda activate base && \
#     conda create -n CatchNet python=3.7 -y && \
#     conda activate CatchNet && \
#     conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11 -c pytorch -c conda-forge -y && \
#     conda install av -c conda-forge -y && \
#     conda install cython -y
    
# # CatchNet runtime environment setup
# RUN mkdir -p /home/i-mind && cd /home/i-mind && \
#     git clone https://github.com/DrMaemi/CatchNet.git && cd CatchNet && \
#     pip install -e . && \
#     pip install -r requirements.txt
