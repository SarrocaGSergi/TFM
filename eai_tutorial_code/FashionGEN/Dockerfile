FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
RUN pip install tensorboard
RUN pip install transformers
COPY requirements.txt /home/sergi/Escritorio/eai_tutorial_code/FashionGEn/
RUN pip install --no-cache-dir -r /home/sergi/Escritorio/eai_tutorial_code/FashionGEn/requirements.txt
COPY utils.py .
COPY FID.py .
COPY models.py .
COPY CWGAN-GP.py .
CMD ["sh", "-c", "umask 0002; python CWGAN-GP.py"]