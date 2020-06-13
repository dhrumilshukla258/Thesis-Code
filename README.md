# Clustering analysis of multi-channel microwave satellite imagery to classify tropical cyclone intensity

The repository consist of Code I wrote while working on my thesis.

To recreate the results:

Install Anaconda: https://docs.anaconda.com/anaconda/install/

Libraries used:
Scipy: Reads matfile and provides interpolate function
Matplotlib and Cartopy: For plotting and creating worldmap
Pandas: Reading & Writing csv and text files. Quering tables
sklearn: Clustering Algorithms

Commands to Run the Code:
-> First Creating an Environment
conda create --name thesis
conda activate thesis

-> Installing Necessary Library
conda install -c conda-forge cartopy // Installing cartopy will install all the dependant libaries
conda install -c conda-forge pandas
conda install -c anaconda scikit-learn
conda install -c conda-forge cv2
