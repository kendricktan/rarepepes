# rarepepes
### Make your own rare pepes (inspired by [cyclegan-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix))

![Sample](https://i.imgur.com/O5MUs2h.png)

# Running locally
1. Download [pre-trained weights](https://www.dropbox.com/s/kxuz0ge75e9fsyx/rarepepes_checkpoints.tar.gz?dl=0) and extract it somewhere
2. Setup env
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x conda.sh
./conda.sh  # Remember to add conda to the PATH variable
pip install flask scikit-image
conda install pytorch torchvision -c soumith
```
3. Run server with:
```
python app.py --checkpoints_dir PRE_TRAINED_WEIGHTS_LOCATION
```

# Dataset
The pepe dataset can be downloaded from [dropbox](https://www.dropbox.com/s/mgqiqermp0o9uzp/rarepepes_data.tar.gz?dl=0)

# References
1. [pix2pix](https://arxiv.org/abs/1611.07004)
