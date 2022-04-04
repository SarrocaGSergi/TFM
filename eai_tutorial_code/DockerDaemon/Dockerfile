FROM continuumio/miniconda3

# TODO Here, install your dependencies for example, Pytorch and jupyter.
WORKDIR /app
RUN conda install python=3.8
RUN conda update pip -y \
    && conda install poetry -y \
    && conda install wrapt==1.12.1 llvmlite==0.31.0 pyyaml==5.3 notebook --channel conda-forge -y \
    && conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch \
    && conda install -c huggingface transformers \
    && conda install -c conda-forge tensorboard \
    && conda install -c conda-forge matplotlib \
    && conda install -c anaconda h5py \
    && conda install -c anaconda scipy



# SSH server related changes
# TODO exposed ports, need to fix
EXPOSE 2222
EXPOSE 6000
EXPOSE 8088
ENV LANG=en_US.UTF-8

RUN apt update && \
    apt install -y \
        ca-certificates supervisor openssh-server bash ssh \
        curl wget vim procps htop locales nano man && \
    sed -i "s/# en_US.UTF-8/en_US.UTF-8/" /etc/locale.gen && \
    locale-gen && \
    useradd -m -u 13011 -s /bin/bash toolkit && \
    passwd -d toolkit && \
    useradd -m -u 13011 -s /bin/bash --non-unique console && \
    passwd -d console && \
    useradd -m -u 13011 -s /bin/bash --non-unique _toolchain && \
    passwd -d _toolchain && \
    useradd -m -u 13011 -s /bin/bash --non-unique coder && \
    passwd -d coder && \
    chown -R toolkit:toolkit /run /etc/shadow && \
    apt autoremove --purge && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    echo ssh >> /etc/securetty && \
    rm -f /etc/legal /etc/motd

COPY --chown=13011:13011 --from=registry.console.elementai.com/shared.image/sshd:base /tk /tk

ENTRYPOINT ["/tk/bin/start.sh"]