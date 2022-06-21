# hash:sha256:d1a51bca75fa18ade64ecc1a56b0acc724667b8e4e7b34d413ec58b511801aa8
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        joblib==1.1.0 \
        matplotlib==3.4.3 \
        numpy==1.21.5 \
        pandas==1.3.4 \
        scikit-learn==1.0.1 \
        scipy==1.7.1 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    autograd==1.3 \
    torch==1.10.1

COPY postInstall /
RUN /postInstall
