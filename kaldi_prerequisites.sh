#!/bin/bash

sudo apt-get -y install build-essential
sudo apt-get -y install git
git clone https://github.com/kaldi-asr/kaldi.git kaldi-trunk --origin golden
sudo echo "deb http://dk.archive.ubuntu.com/ubuntu/ trusty main universe1
deb http://dk.archive.ubuntu.com/ubuntu/ trusty-updates main universe" >> /etc/apt/sources.list
sudo apt-get -y update

sudo apt-get -y install linux-headers-$(uname -r)
sudo apt-get -y install flac libflac-dev; 
sudo apt-get -y install libatlas*; 
sudo apt-get -y install subversion; 
sudo apt-get -y install speex libspeex-dev; 
sudo apt-get -y install python-numpy swig; 
sudo apt-get -y install gstreamer-1.0 libgstreamer-1.0-dev; 
sudo apt-get -y install libgstreamer-plugins*; 
sudo apt-get -y install python-pip; pip install --upgrade pip; pip install ws4py; pip install tornado==4; 
sudo apt-get -y install python-anyjson; 
sudo apt-get -y install libyaml-dev; pip install pyyaml; 
sudo apt-get -y install libjansson-dev;
sudo apt-get -y install gnome-applets
sudo apt-get -y install sox

# Install cuda following the steps using the following link:  https://devtalk.nvidia.com/default/topic/1045400/cuda-setup-and-installation/cuda-10-installation-problems-on-ubuntu-18-04/
