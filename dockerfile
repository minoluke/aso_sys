# specify base image
FROM continuumio/miniconda3

# set working directory
WORKDIR /workspace

# set conda-forge
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict

#  install required packages
RUN conda install -y python=3.12 \
    tsfresh=0.20.3 \
    imbalanced-learn=0.12.4 \
    scikit-learn=1.5.2 \
    corner=2.2.2 \
    joblib=1.4.2 \
    numpy=1.26.4 \
    pandas=2.2.2 \
    scipy=1.13.1 \
    statsmodels=0.14.2 \
    py-xgboost \
    lightgbm \
    catboost \
    && conda clean -afy

# copy the content of the local src directory to the working directory
COPY modules /workspace/modules
COPY data /workspace/data
COPY main.py /workspace/main.py
COPY plotter.py /workspace/plotter.py
