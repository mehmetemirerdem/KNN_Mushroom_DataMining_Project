import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_curve, roc_auc_score, classification_report

#Veri yükleme
df=pd.read_csv('C:/Users/emire/Downloads/archive/mushrooms.csv')

#Veri satır ve sütun sayısı
df.shape

#Sütunları gösterir
df.columns

#Veri türlerini gösterir
df.dtypes

#Boş verilen olup olmadığı
df.isnull()

#boş veriler varsa toplam sayısı
df.isnull().sum()

#verileri tablo olarak gösterme
df.head()

#verilerin bilgilerini gösterme
df.info()

#verilerin açıklanması
df.describe()

#sütunlardaki benzersiz değerler
for column in df.columns:
    unique_values = df[column].unique()
    print("\n", f"{column}: {unique_values}")

#class sütununu çıkarma
X=df.drop('class', axis=1)

# y değişkenine class sütununu ekleme
y=df['class']

#Verileri tekrardan tabloda gösterme
X.head()

#y verilerini tabloda gösterme
y.head()

#string değerleri sayısal değerlere dönüştürme
label_encoder_x=LabelEncoder()
for col in X.columns:
    X[col]=label_encoder_x.fit_transform(X[col])

label_encoder_y=LabelEncoder()
y = label_encoder_y.fit_transform(y)

# sayısal değerlere değiştikten sonra tekrardan tabloda görüntüleme işlemi
X.head()
y

# x ve y olarak ayrılmış verileri train ve test olarak ayırmak
train_x,test_x,train_y,test_y=train_test_split(X,y)

#Model için en uygun k değerini bulma işlemi
acc = []
for neighbors in range(3,10,1):
    classifier = KNeighborsClassifier(n_neighbors=neighbors, metric='minkowski')
    classifier.fit(train_x,train_y)
    y_pred = classifier.predict(test_x)
    acc.append(accuracy_score(test_y,y_pred))

# grafiği ekrana yazdırma işlemi
plt.figure(figsize=(15,7))
plt.plot(list(range(3,10,1)), acc)
plt.show()

# en uygun değeri yazdırma işlemi
print(f"Best accuracy is {np.max(acc)} and the k value is {1+acc.index(np.max(acc))}")

#En uygun değere göre modeli tekrardan eğitme işlemi
k=1+acc.index(np.max(acc))
knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(train_x,train_y)
pred=knn.predict(test_x)

# hata sonucunu yazdırma işlemi
print("Mean absolute error:",mean_absolute_error(pred,test_y))

#modelin sonuçlarını tutmak için bir array oluşturup bu arraye sonuçları ekleme işlemi
pred2=[]
for i in pred:
    if i==1:
        pred2.append('p')
    else:
        pred2.append('e')

# sonuçları tabloda yazdırma işlemi
pred2=pd.DataFrame({'index':test_x.index,'class':pred2})
pred2.head()

# eğer istenirse sonuçları csv dosyasına yazdırma işlemi
pred2.to_csv('submission.csv',index=False)

# Metrik hesaplamaları
accuracy = accuracy_score(test_y, y_pred)
precision = precision_score(test_y, y_pred)
recall = recall_score(test_y, y_pred)
f1 = f1_score(test_y, y_pred)
report = classification_report(test_y, y_pred)

#yazdırma işlemi
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("Classification Report:", report)

#confusion matrix
KNN_confusionmatrix = confusion_matrix(test_y, y_pred)
plt.figure(figsize=(7,5))
plt.title('Confusion Matrix for KNN Classifier')
sns.heatmap(KNN_confusionmatrix , annot=True,xticklabels=["Edible", "Poisonous"], yticklabels=["Edible", "Poisonous"],fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Correct Label')
plt.show()

print('Confusion matrix Accuracy is: {}'.format(metrics.accuracy_score(test_y, y_pred)))

#Roc-curve grafiğini kullanarak görselleştirme
y_pred_prob = knn.predict_proba(test_x)[:, 1]
fpr, tpr, thresholds = roc_curve(test_y, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

#

y_pred_prob = knn.predict_proba(test_x)[:, 1]

# precision-recall grafiği
precision, recall, thresholds = precision_recall_curve(test_y, y_pred_prob)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()