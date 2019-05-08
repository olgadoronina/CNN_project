# Deconvolution of turbulent flow using Convolutional Neural Networks 

## Intro:
It is computationally expensive to resolve all scales in a turbulent flow simulation (Direct Numerical Simulation (DNS)) 
thus it is common practice to use a cheaper,  coarse-grained simulation such as  Large Eddy Simulation (LES). 
LES uses coarse grids and simulates only coarse-grained (resolved) variables while modeling the subgrid effects. 
The simulated data output by the LES models can be thought of as a filtered velocity field, i.e. the simulation 'smooths' 
out the velocity values and erases small structures. 

The aim of this project is to recover the true structure of a velocity field from its coarse-grained computation using 
an convolutional neural network architecture. 

LES velocity filtering can be modeled as a translation-invariant convolution $x = k*y$, where $y$ is the original field, 
$k$ is the convolutional kernel and $x$ is the resulting filtered velocity field. 
Thus restoration of the original velocity field is an inverse process called deconvolution and can be expressed as 
$y = k^{\dagger}*x$, where k^{\dagger} denotes the pseudo-inverse kernel.

The network architecture is adopted from Xu et al. paper (2014): 
[Deep convolutional neural network for image deconvolution](https://papers.nips.cc/paper/5485-deep-convolutional-neural-network-for-image-deconvolution.pdf), 
where the authors use kernel separability achieved by singular value decomposition (SVD) of the pseudo-inverse kernel 
$k^{\dagger}=USV^T$. This allows them to express a 2D convolution as a weighted sum of separable 1D kernels and avoid 
rapid weight-size expansion. 

## Data
I used turbulence flow data from [Johns Hopkins Turbulence Database (JHTDB)](http://turbulence.pha.jhu.edu/), 
n particular, Isotropic 1024 Coarse dataset. 
To populate dataset for CNN, I used rotation, reflection and shifting (since data is periodic). 
Shuffled ready-to-use dataset can be found [here](https://drive.google.com/drive/folders/1F9qDJkgm9WPUz7wqDB8Oovs-OTkqCm5W?usp=sharing)

## To Run
The main script should be run from the `Github` directory in order to correctly call the data files for reading.  

`Github$ python main.py`

## main.py
Depending on the model, the program is set up to iterate through a different number of epochs and neurons, 
retraining a model for each and outputting separate results.  The following variables are configurable:





