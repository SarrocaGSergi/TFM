FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
RUN pip install tensorboard
RUN pip install transformers
COPY requirements.txt /home/sergi/PycharmProjects/TFM/eai_tutorial_code/FashionGEN/GANs/DCGAN/
RUN pip install --no-cache-dir -r /home/sergi/PycharmProjects/TFM/eai_tutorial_code/FashionGEN/GANs/DCGAN/requirements.txt
COPY utils.py .
COPY FID.py .
COPY models.py .
COPY DCGAN.py .
CMD ["sh", "-c", "umask 0002; python DCGAN.py"]