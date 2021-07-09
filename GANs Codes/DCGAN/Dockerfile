FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
RUN conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
RUN pip install tensorboard
COPY requirements.txt /home/sergi/Escritorio/eai_tutorial_code/GANs/DCGAN/
RUN pip install --no-cache-dir -r /home/sergi/Escritorio/eai_tutorial_code/GANs/DCGAN/requirements.txt
COPY DCGAN.py .
CMD ["sh", "-c", "umask 0002; python DCGAN.py"]