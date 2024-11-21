import os
from os import path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import classification_report, confusion_matrix

import TargetCols
import constants


class Clustering:

    def __init__(self, dataframe, target, normalization_method):
        self.dataframe = dataframe
        self.target = target
        self.normmethod = normalization_method
        self.createScaleFolder()

    def createScaleFolder(self):
        path_ = os.path.join("", self.normmethod)
        if not path.exists(path_):
            os.mkdir(path_)

    def elbowKMeans(self, title, init_range, fin_range):
        wcss = []
        for i in range(init_range, fin_range):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(self.dataframe)
            wcss.append(
                sum(np.min(cdist(self.dataframe, kmeans.cluster_centers_, 'euclidean'), axis=1)) / self.dataframe.shape[
                    0])

        number_clusters = range(init_range, fin_range)
        plt.plot(number_clusters, wcss, marker='X')
        plt.title('The KMeans Elbow method for ' + title)
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig(self.normmethod + "/" + 'elbow_' + title + "_" + self.normmethod + '.png')
        plt.show()

    def showRealClustering(self, target, targetCol, title):
        sns.scatterplot(data=self.dataframe, x="p1", y="p2", hue=target[targetCol].to_list())
        plt.title('Target Clusters ' + title)
        plt.legend()
        plt.savefig(self.normmethod + "/" + "real-clustering-" + self.normmethod + "-" + title + ".png")
        plt.show()

    def execKMeans(self, clusters, title):
        # set and execute KMeans
        kmeans = KMeans(n_clusters=clusters, random_state=0)
        kmeans.fit_predict(self.dataframe)
        # end of execution

        # plot kmeans region
        print("Predicted Clusters: ", str(list(set(kmeans.labels_))))
        if title == TargetCols.nsp_name:
            print("Cluster labeling procedure for KMeans algorithm")
            self.ClusterLabelingProc(kmeans.labels_, 'Kmeans', [0, 1, 2])
        else:
            sns.scatterplot(data=self.dataframe, x="p1", y="p2", hue=kmeans.labels_)
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                        marker="^", c="r", s=80, label="centroids")
            plt.legend()
            plt.savefig(
                self.normmethod + "/" + 'Kmeans_' + str(clusters) + "_" + self.normmethod + "-" + title + ".png")
            plt.title('Predicted Clusters - Kmeans ' + title)
            plt.show()
        # end of plot kmeans region
        return kmeans.labels_

    def HCKmeans(self, clusters, title):
        cluster = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='ward')
        copied_df = self.dataframe.copy()
        cluster.fit_predict(copied_df)
        print("Predicted Clusters: ", str(len(list(set(cluster.labels_)))))
        if title == TargetCols.nsp_name:
            print("Cluster labeling procedure for HCKMeans algorithm")
            self.ClusterLabelingProc(cluster.labels_, 'HCKmeans', [1, 0, 2])
        else:
            sns.scatterplot(data=copied_df, x="p1", y="p2", hue=cluster.labels_)
            plt.title('Predicted Clusters - HCKmeans ' + title + "_" + self.normmethod)
            plt.legend()
            plt.savefig(
                self.normmethod + "/" + 'HCKmeans_' + str(clusters) + "_" + self.normmethod + "_" + title + ".png")
            plt.show()

        return cluster.labels_

    def SpectralClustering(self, cluster, title):
        cf = self.dataframe.copy()
        spectral_model_rbf = SpectralClustering(n_clusters=cluster, random_state=0)
        labels = spectral_model_rbf.fit_predict(cf)
        cf['PredictedLabels'] = labels
        if title == TargetCols.nsp_name:
            print("Cluster labeling procedure for Spectral algorithm")
            self.ClusterLabelingProc(labels, 'Spectral Clustering ' + title, [0, 2, 1])
        else:
            sns.scatterplot(data=cf, x="p1", y="p2", hue=cf['PredictedLabels'])
            plt.title('Predicted Clusters - Spectral-Clustering ' + title + "_" + self.normmethod)
            plt.legend()
            plt.savefig(self.normmethod + "/" + "Spectral-Clustering-" + self.normmethod + "-" + title + ".png")
            plt.show()

    def ClusterLabelingProc(self, labels, cl_algo, labeling_arr):
        parent_dir = constants.path

        colors = np.array(["Red", "Green", "Blue"])
        target = [x - 1 for x in self.target[TargetCols.nsp_name].to_list()]

        # ----------------------------------
        plt.show()
        col_array = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 2, 0],
            [1, 0, 2],
            [2, 1, 0],
            [2, 0, 1]
        ]
        path_ = os.path.join(parent_dir, cl_algo + " " + self.normmethod + "Cl.-Labeling")
        if not path.exists(path_):
            os.mkdir(path_)

        for ar in col_array:
            self.cluster_labeling_body(cl_algo, colors, labels, target, ar, path_, False, True, True)
        # self.cluster_labeling_body(cl_algo, colors, labels, target, labeling_arr, path_, True, True, False)

    def cluster_labeling_body(self, cl_algo, colors, labels, target, test_array, path_, show, print_data, save):
        ar = test_array

        relabel = np.choose(labels, ar).astype(np.int64)
        if print_data:
            print("==========", "Array->", ar, "==========")
            print(classification_report(target, relabel))

        fig, ax = plt.subplots(3)
        fig.suptitle(cl_algo)

        ax[0].scatter(data=self.dataframe, x="p1", y="p2", c=colors[target])
        ax[0].set_ylabel("Target Clusters")
        # ----------------------------------
        ax[1].scatter(data=self.dataframe, x="p1", y="p2", c=colors[relabel])
        ax[1].set_ylabel("Predicted Clusters")
        cm = confusion_matrix(target, relabel)

        ax[2].imshow(cm, interpolation='none', cmap='Blues')
        for (i, j), z in np.ndenumerate(cm):
            ax[2].text(j, i, z, ha='center', va='center')
        ax[2].set_xlabel("Confusion/Similarity Matrix")
        fig.tight_layout()
        if save:
            plt.savefig(path_ + "/" + self.listToString(ar))
        plt.show()

    def listToString(self, s):

        # initialize an empty string
        str1 = ""
        # traverse in the string
        for ele in s:
            str1 += str(ele)

            # return string
        return str1
