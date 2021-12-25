FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
RUN pip install tensorboard
RUN pip install transformers
COPY requirements.txt /home/sergi/Escritorio/eai_tutorial_code/GANs/Transformer-GAN/
RUN pip install --no-cache-dir -r /home/sergi/Escritorio/eai_tutorial_code/GANs/Transformer-GAN/requirements.txt
COPY utils.py .
COPY Tr-GAN.py .
CMD ["sh", "-c", "umask 0002; python Tr-GAN.py"]