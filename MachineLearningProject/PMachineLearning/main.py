import TargetCols as tc
import constants as c
from Clustering import Clustering
from DatasetHandler import Preprocessing
from Classification import Classification


def clusteringNSP(minmax):  # boolean param to use or not minmax scaler (if not uses standard scaler)
    ds1 = Preprocessing(c.path + "CTG.xls", minmax)
    tar1 = ds1.readTargetCol(tc.nsp_unnamed, tc.nsp_col, tc.nsp_enum, ds1.returnKeepRows())

    cl_1 = Clustering(ds1.dataframe, tar1, ds1.getNormMethod())
    cl_1.elbowKMeans(tc.nsp_name, 1, 5)

    cl_2 = Clustering(ds1.dataframe, tar1, ds1.getNormMethod())
    cl_2.execKMeans(3, tc.nsp_name)  # 0,1,2

    cl_3 = Clustering(ds1.dataframe, tar1, ds1.getNormMethod())
    cl_3.HCKmeans(3, tc.nsp_name)  # 1,0,2

    cl_4 = Clustering(ds1.dataframe, tar1, ds1.getNormMethod())
    cl_4.showRealClustering(tar1, tc.nsp_col, tc.nsp_name)

    cl_5 = Clustering(ds1.dataframe, tar1, ds1.getNormMethod())
    cl_5.SpectralClustering(3, tc.nsp_name)  # 0,2,1


def clusteringFHR(minmax):  # boolean param to use or not minmax scaler (if not then uses standard scaler)
    ds = Preprocessing("CTG.xls", minmax)
    tar = ds.readTargetCol(tc.fhr_unnamed, tc.fhr_col, tc.fhr_enum, ds.returnKeepRows())

    cl = Clustering(ds.dataframe, tar, ds.getNormMethod())
    cl.elbowKMeans(tc.fhr_name, 6, 16)

    cl1 = Clustering(ds.dataframe, tar, ds.getNormMethod())
    cl1.execKMeans(10, tc.fhr_name)

    cl2 = Clustering(ds.dataframe, tar, ds.getNormMethod())
    cl2.HCKmeans(10, tc.fhr_name)

    cl3 = Clustering(ds.dataframe, tar, ds.getNormMethod())
    cl3.SpectralClustering(10, tc.fhr_name)

    cl4 = Clustering(ds.dataframe, tar, ds.getNormMethod())
    cl4.showRealClustering(tar, tc.fhr_col, tc.fhr_name)


if __name__ == '__main__':
    # ===========Preprocessing===========
    ds_minmax = Preprocessing("CTG.xls", True).dataframe
    print(ds_minmax)
    ds_std = Preprocessing("CTG.xls", False).dataframe
    print(ds_std)
    # ===========Clustering===========
    # ===========clustering - FHR===========
    clusteringFHR(True)#min-max scaler
    clusteringFHR(False)#standard scaler
    # ===========clustering - NSP===========
    clusteringNSP(True)#min-max scaler
    clusteringNSP(False)#standard scaler
    # ======================================
    # =========== Classification ===========
    cls1 = Classification(1, 100, 8, True)  # min-max scaler
    cls2 = Classification(2, 50, 4, False)  # standard scaler
    cls3 = Classification(1, 120, 32, False)  # standard scaler
    cls4 = Classification(2, 110, 64, True)  # min-max scaler

