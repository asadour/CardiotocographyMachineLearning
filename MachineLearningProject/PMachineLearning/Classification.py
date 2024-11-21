import os
from time import time
from os import path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

import TargetCols
from DatasetHandler import Preprocessing


def train_evaluate(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


class Classification:
    def __init__(self, mode, epochs, batch_size, scale_type):
        self.model_list = []
        self.mode = mode
        self.batch_size = batch_size
        self.epochs = epochs
        self.scaling_type = scale_type
        self.pathFolder = ""
        self.createScaleFolder()
        self.runClassification()

    def createScaleFolder(self):
        path_ = os.path.join("", "Classification_"+str(self.batch_size)+"_"+str(self.epochs))
        self.pathFolder = str(path_)
        if not path.exists(path_):
            os.mkdir(path_)

    def runClassification(self):
        prep = Preprocessing('CTG.xls', self.scaling_type)  # read preprocessed dataset
        X = prep.scaling(prep.main_dataset)
        y = prep.readTargetCol(TargetCols.fhr_unnamed, TargetCols.fhr_col, TargetCols.fhr_enum, prep.returnKeepRows())
        input_shape = X.shape[1]
        output_shape = len(y['CLASS'].unique()) + 1

        self.runCV(X, y, input_shape, output_shape)

    def createModel(self, input_shape, output_shape):
        optimizer = Adam(learning_rate=3e-4, decay=1e-4)

        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(input_shape,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(output_shape, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        return model

    def createDynamicModel(self, input_shape, output_shape, input_layer_size, denses_list):
        optimizer = Adam(learning_rate=3e-4, decay=1e-4)
        model = Sequential()
        model.add(Dense(input_layer_size, activation='relu', input_shape=(input_shape,)))
        for dense in denses_list:
            model.add(dense)
        model.add(Dense(output_shape, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        return model

    def runModel(self, input_shape, output_shape, x_train, y_train, x_val, y_val,
                 fold_no=0):  # mode=1 or 2 changes the model that uses
        # ===============================
        if self.mode == 1:
            model = self.createModel(input_shape, output_shape)
        else:
            dense_list = list()
            dense_list.append(Dense(512, activation='relu'))
            dense_list.append(Dense(1024, activation='relu'))
            model = self.createDynamicModel(input_shape, output_shape, 64, dense_list)

        save_model_name = os.path.join('model_classifier' + str(fold_no) + '.h5')
        saveBest = ModelCheckpoint(save_model_name, monitor='val_loss', save_best_only=True)
        earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, verbose=1, mode='min')
        st = time()
        stats = model.fit(x=x_train, y=y_train, batch_size=self.batch_size, epochs=self.epochs,
                          validation_data=(x_val, y_val), callbacks=[saveBest, earlyStopping])
        model.load_weights(save_model_name)
        print('\nTraining duration: {} sec'.format(time() - st))
        self.model_list.append(model)
        return min(stats.history['val_loss']), stats.history['val_loss'], stats.history['loss']

    def runCV(self, X, Y, input_shape, output_shape):  # we run 5-fold cross validation
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        kFold = KFold(5, shuffle=False)
        fold_no = 1
        min_val_losses = []
        losses = []
        val_losses = []
        for train, validation in kFold.split(x_train, y_train):
            x_tr = self.createNewDF(X, train)
            y_tr = self.createNewDF(Y, train)
            x_val = self.createNewDF(X, validation)
            y_val = self.createNewDF(Y, validation)

            y_tr = to_categorical(y_tr)
            y_val = to_categorical(y_val)

            min_val_loss, val_loss, loss = self.runModel(input_shape, output_shape, x_tr, y_tr, x_val, y_val, fold_no)
            min_val_losses.append(min_val_loss)
            losses.append(loss)
            val_losses.append(val_loss)

            fold_no += 1

        print(val_losses, val_losses.index(min(val_losses)))

        min_loss_idx = val_losses.index(min(val_losses))
        model = self.model_list[min_loss_idx]

        y_preds = model.predict(x_test)
        y_preds = np.argmax(y_preds, axis=1)
        print("Run for ", self.epochs, " epochs and batch_size=", self.batch_size)
        print("===================================================================")
        print("~~~Confusion Matrix~~~")
        print(confusion_matrix(y_test, y_preds))
        print('Classification report of model on test data -> \n',
              classification_report(y_test, y_preds, zero_division=1))

        # =============== loss plot =====================
        epochs = [i for i in range(1, len(losses[min_loss_idx]) + 1)]
        plt.plot(epochs, losses[min_loss_idx], label='Training')
        plt.plot(epochs, val_losses[min_loss_idx], label='Validation')
        plt.title('Loss vs. Epochs')
        plt.savefig(self.pathFolder+"/"+"bestmodel_losses.png")
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()

    def createNewDF(self, old_df, rows_stay):
        olddatafr = old_df.copy()
        rows = olddatafr.iloc[rows_stay, :]
        print(rows)
        return rows
