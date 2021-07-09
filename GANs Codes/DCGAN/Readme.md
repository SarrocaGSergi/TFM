# DCGAN EAI Code

## How to execute:

### Build docker image: 

docker build -t registry.console.elementai.com/uab.uab_tfm/dcgan_test .

docker push registry.console.elementai.com/uab.uab_tfm/dcgan_test

### Launch Jobs: 

eai job new \
--data uab.uab_tfm.mnist:/app/data/mnist \
--data uab.uab_tfm/test_results:/app/test_results \
--image registry.console.elementai.com/uab.uab_tfm/dcgan_test \
--gpu 1 --mem 16

eai job new \
--data uab.uab_tfm/test_results:/app/test_results \
--image tensorflow/tensorflow \
-- tensorboard --bind_all --log_dir /app/test_results
