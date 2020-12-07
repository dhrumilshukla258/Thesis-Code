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

### Before creating images:
Go through the Examples\mat_to_img (full process example).pynb <br/>
It explains on how the images are created and what changes can be made to create image with or without prerequisite. Additionally, you want to add or remove the colorbar or the axis, then you can edit the code in the function __ReadFreq of FreqReader Class provided in the mat_to_img (full process example).pynb file.

To implement multiprocessing, I wrote scripts to create images. Go through the Read_Storm folder for that. <br/>
In the scripts I have commented the code of adding axis and colorbar to the image, to uncomment go through the ReadFileAndCreateImages function of matreader class in the matfile_reader.py file. The structre of code is similar to how its written in pynb file.

### To create images from the .mat files
```
cd Read_Storm
python multiprocess_image_creation.py
```

### Applying Clustering
Go through the Examples\cluster_test.ipynb. It provides the deatils on how the cluster algorithms are applied, octant creation and the test data selection. The structure of the code is similar to how it's written in the ipynb file. When we apply multiprocessing on the clustering algorithms each of the process which uses AllPixels selection requres more than 2-3GB of ram. This results in the CPU getting crashed. Due to this the cluster algorithm doesn't use the multiprocessing.

### To apply cluster algorithms on the images
```
cd Apply_Clustering
python cluster_iteration.py
```

### Applying Clustering Analysis
The Apply_Cluster_Analysis\test_result_script.py, consist of class of methods consisting of various analysis applied on all cluster methods.

### To apply cluster analysis on the cluster output
```
cd Apply_Cluster_Analysis
python test_iteration_script.py
```