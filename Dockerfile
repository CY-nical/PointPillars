FROM python:3.12-bookworm

COPY requirements.txt /pointpillars_ml/
COPY figures /pointpillars_ml/
COPY misc /pointpillars_ml/
COPY pointpillars /pointpillars_ml/
COPY pretrained /pointpillars_ml/
COPY evaluate.py /pointpillars_ml/
COPY pre_precess_kitti /pointpillars_ml/
COPY setup.py /pointpillars_ml/
COPY test.py /pointpillars_ml/
COPY train.py /pointpillars_ml/

WORKDIR /pointpillars_ml/


RUN pip install -r requirements.txt