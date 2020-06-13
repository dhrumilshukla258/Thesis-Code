# Clustering analysis of multi-channel microwave satellite imagery to classify tropical cyclone intensity

The repository consist of Code I wrote while working on my thesis.

To recreate the results:

Install Anaconda: https://docs.anaconda.com/anaconda/install/

Libraries not included in anaconda which were used in the thesis:
Scipy: Reads matfile and provides interpolation function | Agglomerative clustering (Linkage Method) | Dendrogram Analysis
Matplotlib and Cartopy: For plotting and creating worldmap
Pandas: Reading & Writing csv and text files. Quering tables
sklearn: Kmeans Algorithm
OpenCV: Reads Images

Commands to Install Libraries:
-> Frist Creating an Environment
conda create --name thesis
conda activate thesis

-> Installing Necessary Library
conda install -c conda-forge pandas
conda install -c conda-forge opencv
conda install -c conda-forge cartopy // Installing cartopy will install all the dependant libaries
conda install -c conda-forge notebook
conda install -c anaconda scikit-learn
