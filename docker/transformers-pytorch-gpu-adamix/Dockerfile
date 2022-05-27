FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
FROM yaqing/pytorch-efficient-tuning:v0.1
LABEL maintainer="Hugging Face"
LABEL repository="transformers"





COPY . transformers/
RUN cd transformers/ && \
    python -m pip install --no-cache-dir .




CMD ["/bin/bash"]

