# impor library panda untuk membaca data
import pandas
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import warnings
warnings.filterwarnings('ignore')

# membaca file dan direktori tempat nilai_mhs1.csv disimpan
direktori = "titanic_survivor.csv"
var = ['Passengerid','Age','Fare','Sex','sibsp','zero1','zero2','zero3','zero4','zero5',
       'zero6','zero7','Parch','zero8','zero9','zero10','zero11','zero12','zero13','zero14','zero15',
       'Pclass','zero16','zero17','Embarked','zero18','zero19','2urvived']

# membaca data dengan library panda
data = pandas.read_csv(direktori, names=var)

# mengisi data yang ksoong dengan mean
data = data.fillna(data.mean())
data.isna().sum()

# pisah fitur dengan kelas 
import numpy as np
x = np.array(data[['Passengerid','Age','Fare','Sex','sibsp', 'Parch','Pclass','Embarked']])
Y = np.array(data['2urvived'])

# buat pembagian data test dan train dengan k-fold CV
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

# Support Vector Machine
from sklearn.svm import SVC
# matric untuk menghitung akurasi dan mendapatkan confussion matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score

total_akurasi_arr1 = list()

total_akurasi_arr2 = list()

total_akurasi_arr3 = list()

# jumalh kernel yg dipakai
k = ["sigmoid", "poly", "rbf"]
a = 0
for j in k:
    ulang = 5
    for i in range(ulang):
        akurasi_arr = list()

        for train, test in skf.split(x,Y):
            X_train, X_test, y_train, y_test = x[train], x[test], Y[train], Y[test]
            # fit SVM model ke data train
            model = SVC(kernel=k[a])
            model.fit(X_train, y_train)
            # model yang terbentuk
            print(model)
            # buat prediksi dengan data test
            expected = y_test
            predicted = model.predict(X_test)
            # ringkasan hasil evaluasi
            print(metrics.classification_report(expected, predicted))
            print(metrics.confusion_matrix(expected, predicted))
            print()

            akurasi = accuracy_score(expected, predicted)
    
            akurasi_arr.append(akurasi)
    
            print('Akurasi = ', round(akurasi, 2))
            print()
            print()

        print('total akurasi = ', round((sum(akurasi_arr)/5), 2))
        print()
        print()

        if a == 0:
            total_akurasi_arr1.append(round((sum(akurasi_arr)/5), 2))
        elif a == 1 :
            total_akurasi_arr2.append(round((sum(akurasi_arr)/5), 2))
        else :
            total_akurasi_arr3.append(round((sum(akurasi_arr)/5), 2))

    a += 1


print('Array looping 5X total Akurasi ', k[0], ' = ', total_akurasi_arr1)
print('Array looping 5X total Akurasi ', k[1], ' = ', total_akurasi_arr2)
print('Array looping 5X total Akurasi ', k[2], ' = ', total_akurasi_arr3)

# menggambar grafik plot akurasi     
plt.figure(figsize = (13, 7))
plt.plot(np.arange(1,6,1), total_akurasi_arr1, marker = "o", linewidth=1, label=k[0])
plt.plot(np.arange(1,6,1), total_akurasi_arr2, marker = "o", linewidth=1, label=k[1])
plt.plot(np.arange(1,6,1), total_akurasi_arr3, marker = "o", linewidth=1, label=k[2])
# plt.bar(np.arange(1,4,1), akurasi_arr_grafik)
plt.xlabel('Jumlah Looping')
plt.title("SVM Grafik akurasi kernel")
plt.legend()
plt.ylim(0, 1)
plt.show()