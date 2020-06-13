# Clustering analysis of multi-channel microwave satellite imagery to classify tropical cyclone intensity

The repository consist of Code I wrote while working on my thesis.

To recreate the results:

Install Anaconda: https://docs.anaconda.com/anaconda/install/

### Libraries not included in anaconda which were used in the thesis:
**Scipy**: Reads matfile | Provides interpolation function | Agglomerative clustering (Linkage Method) | Dendrogram Analysis <br />
**Matplotlib and Cartopy**: Plotting and creating worldmap <br />
**Pandas**: Reading & Writing csv and text files | Quering tables <br />
**sklearn**: Kmeans Algorithm <br />
**OpenCV**: Read Images

### Install necessary Libraries via Anaconda Command Prompt:
-> Frist create an Environment
```
conda create --name thesis
conda activate thesis
```
-> Installing Libraries <br />
```
conda install -c conda-forge pandas
conda install -c conda-forge opencv
```
Installing cartopy will install all the necessary libaries
```
conda install -c conda-forge cartopy
conda install -c conda-forge notebook
conda install -c anaconda scikit-learn
```
