FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
RUN pip install tensorboard
RUN pip install transformers
COPY requirements.txt /home/sergi/PycharmProjects/TFM/eai_tutorial_code/FashionGEN/ConditionalBatchNorm/CBN-WGANGP/
RUN pip install --no-cache-dir -r /home/sergi/PycharmProjects/TFM/eai_tutorial_code/FashionGEN/ConditionalBatchNorm/T-WGANGP/requirements.txt
COPY utils.py .
COPY FID.py .
COPY Dataset.py .
COPY CondBatchNorm.py .
COPY WGAN_Arch.py .
COPY CBN-WGANGP.py .
CMD ["sh", "-c", "umask 0002; python CBN-WGANGP.py"]