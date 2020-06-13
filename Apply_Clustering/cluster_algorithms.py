from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

import numpy as np
import random
import json
import copy
import math

def FuzzySimilarity(x,c):
    #x [ 1, 2 ,3 ,4 ] c[ 0 ,1 ,2 ,3]    
    A = np.minimum(x,c)
    B = np.maximum(x,c)
    return (np.sum(A) + 1) / (np.sum(B) + 2)

class Clustering():
    def __init__(self,x,clusterSize=2,max_iter=300):
        self.k = clusterSize
        self.m = x.shape[0]
        self.p = x.shape[1]
        self.x = x
        max_x = np.amax(self.x, axis=0)
        max_x[max_x == 0] = 1
        self.norm_x = self.x / max_x
        self.max_iter = max_iter
        self.n_iter = 0
        self.labels = []
        self.cluster_centers = np.zeros((self.k,self.p))

    def Fuzzy_K_Means(self,w1=1.0,w2=1.0,w3=1.0):
        prevC = np.random.rand(self.k,self.p)
        rand_center_index = random.sample(range(0, self.m), self.k)
        
        # Randomly assigning each cluster center
        # a unique x (whose index lies from 0 to m)
        for j in range(self.k):
            self.cluster_centers[j,:] = self.norm_x[rand_center_index[j],:] 
        
        for i in range(self.m):
            self.labels.append(0)

        isConverged = False
        while not isConverged and self.n_iter<self.max_iter:
            #Assigning each images, cluster where it belongs 
            for i in range(self.m):    
                maxSimilarity = -math.inf
                index = -1
                for j in range(self.k):
                    #Calculating Similarity for Image, Pressure and Wind
                    #s1 = FuzzySimilarity(self.norm_x[i,0:self.p-2],self.cluster_centers[j,0:self.p-2])
                    #s2 = FuzzySimilarity(self.norm_x[i,self.p-2:self.p-1],self.cluster_centers[j,self.p-2:self.p-1])
                    #s3 = FuzzySimilarity(self.norm_x[i,self.p-1:self.p],self.cluster_centers[j,self.p-1:self.p])
                    #fs = ( (w1*s1) + (w2*s2) + (w3*s3) ) / (w1+w2+w3)
                    fs = FuzzySimilarity(self.norm_x[i,:],self.cluster_centers[j,:])
                    if maxSimilarity < fs:
                        maxSimilarity = fs
                        index = j

                # Assigning cluster(1...k) to each images based on MaxSimilarity value
                self.labels[i] = index


            # Iterating through each images
            # Each labels determine unique cluster, images are assigned to
            total = np.zeros((self.k,self.p+1))
            for i in range(self.m):
                total[self.labels[i],0] += 1 
                total[self.labels[i],1:] += self.norm_x[i,:]

            # Updating cluster center            
            for j in range(self.k):
                self.cluster_centers[j,:] = total[j,1:]/ (total[j,0])
                
            if (prevC == self.cluster_centers).all():
                isConverged = True

            self.n_iter+=1     
            prevC = copy.deepcopy(self.cluster_centers)
            
        self.cluster_centers *= np.amax(self.x, axis=0)
        self.silhouetteAvg = silhouette_score(self.x, self.labels)
        self.silhouetteValues = silhouette_samples(self.x, self.labels)
    
    def Scikit_K_Means(self):
        kmeans = KMeans(n_clusters=self.k,max_iter=self.max_iter).fit(self.x)
        self.cluster_centers = kmeans.cluster_centers_
        self.n_iter = kmeans.n_iter_
        self.labels = kmeans.labels_
        self.silhouetteAvg = silhouette_score(self.x, self.labels)
        self.silhouetteValues = silhouette_samples(self.x, self.labels)
        
    def __fancy_dendrogram(self, *args, **kwargs):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)

        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            plt.title('Hierarchical Clustering Dendrogram (truncated)')
            plt.xlabel('sample index or (cluster size)')
            plt.ylabel('distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%0.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
        return ddata
    
    
    def Scipy_Agglomerative(self, dist_m, link_m, path):
        Z = linkage(self.x,method=link_m,metric=dist_m)
        label_ans = fcluster(Z, t=int(self.k), criterion='maxclust')
        fig, axes= plt.subplots(nrows=1, ncols=1,figsize=(30,15))
        
        dn = self.__fancy_dendrogram(Z, 
                        labels=label_ans, 
                        ax=axes, 
                        truncate_mode='lastp', 
                        p=int(self.k), 
                        orientation='top',
                        get_leaves = True,
                        annotate_above=10,
                        show_leaf_counts = True,
                        leaf_rotation = 90,
                        leaf_font_size = 20
                       )
        
        with open(path+"dendogram_lastp_"+str(self.k)+".json", 'w') as f:
            json.dump(dn, f)
        
        plt.savefig(path+"dendrogram_lastp_"+str(self.k)+".png",format="png")
        
        #dn = dendrogram(Z, labels=label_ans, ax=axes, truncate_mode='level', p=i, orientation='top')
        #plt.savefig(path+"dendrogram_level_"+str(i)+".png",format="png")
        
        plt.close()
        
        self.labels = label_ans
        self.labels = [x - 1 for x in self.labels]
        self.silhouetteAvg = silhouette_score(self.x, self.labels)
        self.silhouetteValues = silhouette_samples(self.x, self.labels)