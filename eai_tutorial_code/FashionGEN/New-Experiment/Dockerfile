FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
RUN pip install tensorboard
RUN pip install transformers
COPY requirements.txt /home/sergi/Escritorio/TFM-main/eai_tutorial_code/FashionGEN/New-Experiment/
RUN pip install --no-cache-dir -r /home/sergi/Escritorio/TFM-main/eai_tutorial_code/FashionGEN/New-Experiment/requirements.txt
COPY utils.py .
COPY FID.py .
COPY models.py .
COPY CustomCBN.py .
COPY TestGAN.py .
CMD ["sh", "-c", "umask 0002; python TestGAN.py"]