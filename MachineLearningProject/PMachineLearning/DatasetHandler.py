import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import constants


class Preprocessing:

    def __init__(self, filename, minmaxscaler):
        self.name = filename
        self.dataframe = None
        self.target = None
        self.initialDatafr = None
        print("Preprocessing started....")
        self.minmax = minmaxscaler
        self.keep = []
        self.main_dataset = None
        self.readDataSet()

    def returnKeepRows(self):
        return self.keep

    def readTargetCol(self, unnamed_column, column_name, col_enum, keep_rows):
        f = pd.read_excel(self.name, sheet_name=2, header=1, usecols=col_enum, skipfooter=3)
        f.rename(columns={unnamed_column: column_name}, inplace=True)
        print(f.head(10))

        print(f.isnull().any().any())
        print("No of null cols ", f.isnull().sum().sum())
        pf = f.dropna()
        pf = self.createNewDataset(pf, keep_rows)
        self.target = pf
        print("(Data Rows: ", str(self.dataframe.shape[0]), " Target Rows: ", str(self.target.shape[0]) + ")")
        print(self.initialDatafr, "\n", self.target)
        return pf

    def readDataSet(self):
        f = pd.read_excel(self.name, sheet_name=2, header=1, usecols="G:AB", skipfooter=3)
        f.rename(
            columns={'Unnamed: 6': 'LB', 'Unnamed: 7': 'AC', 'Unnamed: 8': 'FM', 'Unnamed: 9': 'UC',
                     'Unnamed: 10': 'ASTV',
                     'Unnamed: 11': 'MSTV', 'Unnamed: 12': 'ALTV', 'Unnamed: 13': 'MLTV', 'Unnamed: 14': 'DL',
                     'Unnamed: 15': 'DS', 'Unnamed: 16': 'DP', 'Unnamed: 17': 'DR', 'Unnamed: 18': 'Width',
                     'Unnamed: 19': 'Min', 'Unnamed: 20': 'Max', 'Unnamed: 21': 'Nmax', 'Unnamed: 22': 'Nzeros',
                     'Unnamed: 23': 'Mode', 'Unnamed: 24': 'Mean', 'Unnamed: 25': 'Median', 'Unnamed: 26': 'Variance',
                     'Unnamed: 27': 'Tendency'}, inplace=True)

        f = self.columnValidation(f)
        f = f.dropna()
        f = f.drop(['DR'], axis=1)#den perilamvanetai sta xaraktiristika
        self.dataframe = f
        self.initialDatafr = f
        self.dataframe = self.scaling_and_pca()
        self.main_dataset = f
        return f

    def columnValidation(self, df):
        removeRowsList = []

        for index, row in df.iterrows():
            if row['ASTV'] >= 0 or row['ASTV'] <= 100:
                pass
            else:
                removeRowsList.append(index)
            if row['ALTV'] >= 0 or row['ALTV'] <= 100:
                pass
            else:
                removeRowsList.append(index)
            if row['Min'] >= 0 and row['Max'] >= 0 and row['Nmax'] >= 0 and row['Nzeros'] >= 0 and row['Width'] >= 0 and \
                    row['DP'] >= 0 and row['DS'] >= 0 and row['DL'] >= 0:
                pass
            else:
                removeRowsList.append(index)
            if row['Tendency'] == -1 or row['Tendency'] == 0 or row['Tendency'] == 1:
                pass
            else:
                removeRowsList.append(index)
            # katharismos twn dedomenwn meta ti siblirosi tis listas
        delete_rows = list(set(removeRowsList))
        print("Old Data", removeRowsList)
        print("Fault Data", np.asarray(delete_rows))
        print("Dataset size before: ", df.shape[0])
        keep_rows = []
        for row in range(df.shape[0]):
            if row not in delete_rows:
                keep_rows.append(row)

        df = self.createNewDataset(df, keep_rows)
        self.keep = keep_rows
        print(self.keep)
        print("Dataset size after: ", df.shape[0])
        return df

    def createNewDataset(self, old_df, rows_stay):
        olddatafr = old_df.copy()
        rows = olddatafr.iloc[rows_stay, :]
        return rows

    def scaling(self, df):
        # MinMaxScaler/StandardScaler region
        if self.minmax is True:
            mmScaler = MinMaxScaler()
        else:
            mmScaler = StandardScaler()
        scaled = mmScaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled, columns=df.columns)
        return scaled_df

    def scaling_and_pca(self):
        # print(scaled_features)
        print("Scaling and PCA procedures started...")
        # MinMaxScaler/StandardScaler region
        if self.minmax is True:
            mmScaler = MinMaxScaler()
        else:
            mmScaler = StandardScaler()
        scaled = mmScaler.fit_transform(self.dataframe)
        print(self.getNormMethod(), " => ", scaled)
        scaled_df = pd.DataFrame(scaled, columns=self.dataframe.columns)
        # end of MinMaxScaler/StandardScaler

        # PCA region
        df = scaled_df
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(df)
        principalDf = pd.DataFrame(data=principalComponents, columns=['p1', 'p2'])
        # end of PCA region

        # print(principalDf)
        self.dataframe = principalDf

        return principalDf

    def getNormMethod(self):
        if self.minmax:
            return constants.minmax
        else:
            return constants.standard
