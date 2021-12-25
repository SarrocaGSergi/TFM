FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
RUN pip install tensorboard
COPY requirements.txt /home/sergi/Escritorio/eai_tutorial_code/GANs/WGAN-GP/
RUN pip install --no-cache-dir -r /home/sergi/Escritorio/eai_tutorial_code/GANs/WGAN-GP/requirements.txt
COPY utils.py .
COPY WGAN-GP.py .
CMD ["sh", "-c", "umask 0002; python WGAN-GP.py"]