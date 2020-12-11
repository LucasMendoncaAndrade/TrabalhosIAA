"""
Trabalho integrador
07/12/2020
Diagnóstico de pneumonia a partir de raio X
Utilizou-se convolutional neural network (CNN)

Hugo Felipe Ferreira
Lucas Mendonça Andrade
Vitor Fernandes Azevedo
"""

#Bibliotecas a serem utilizadas
import pandas as pd
import numpy as np
import os, cv2, random, pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

size = 224
training_data = []

#Criação da função que importa as imagens
def image_import (img_path, class_type, appended_list):
    for i in os.listdir(img_path):
        read_path = os.path.join(img_path, i)
        try:
            img = cv2.imread(read_path)
            img = cv2.resize(img, (size, size))
            appended_list.append([img, class_type])
        except Exception as e:
            print(f'Image error: {i}')
            pass
        
#Importação das imagens de treino da pneumonia
#Coloque o diretório exato das imagens
img_path = 'C:/Users/Lucas/Desktop/Trabalho/archive/chest_xray/train/PNEUMONIA'
image_import(img_path,1, training_data)

#Importação das imagens de treino de pulmões normais
#Coloque o diretório exato das imagens
img_path = 'C:/Users/Lucas/Desktop/Trabalho/archive/chest_xray/train/NORMAL'
image_import(img_path, 0, training_data)

#Embalharamento dos dados de treinamento
random.shuffle(training_data)

#Separação XY
features = []
targets = []

for f, t in training_data:
    features.append(f)
    targets.append(t)

X = np.array(features).reshape(-1, size, size, 3)
Y = np.array(targets)

datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        validation_split=0.2,
        brightness_range=[0.2,1.0],)
test_img = datagen.flow(
    X,
    Y,
    subset="training")
val_img = datagen.flow(
    X,
    Y,
    subset="validation")

#Modelo nomeado de MobileNetV2
base_model=tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling=max,
    classifier_activation="softmax",
)
base_model.trainable = False

#Construção do modelo
model = Sequential()
model.add(base_model)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1))    
model.add(Activation('sigmoid'))
model.summary()
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#Modelo de treino
early_stop = EarlyStopping(monitor = 'val_loss', patience = 2)
model.fit(test_img, batch_size = 32, epochs = 12,validation_data = val_img)

#Salvar modelo
model.save('pneumonia_cnn.model')

#Importação das imagens de teste da pneumonia
#Coloque o diretório exato das imagens
img_path = 'C:/Users/Lucas/Desktop/Trabalho/archive/chest_xray/test/PNEUMONIA'
testing_data=[]
image_import(img_path, 1, testing_data)

#Importação das imagens de teste de pulmões normais
#Coloque o diretório exato das imagens
img_path = 'C:/Users/Lucas/Desktop/Trabalho/archive/chest_xray/test/NORMAL'
image_import(img_path,0, testing_data)

#Embalharamento dos dados de treinamento
random.shuffle(testing_data)

#Separação XY
test_features = []
test_targets = []

for f, t in testing_data:
    test_features.append(f)
    test_targets.append(t)

test_features = np.array(test_features).reshape(-1, size, size, 3)
test_targets = np.array(test_targets)   
test_features = test_features/255

#Cálculo da matriz de confusão
false_negative = 0
false_positive = 0
true_negative = 0
true_positive = 0

result = model.predict_classes(test_features)
accuracy = model.evaluate(test_features,test_targets)[1]

for i in range(len(test_features)):
    if test_targets[i] == 1:
        if result[i] == 1:
            true_positive += 1
        elif result[i] == 0: 
            false_negative += 1
        else:
            print('error class not found')
    if test_targets[i] == 0:
        if result[i] == 1:
            false_positive += 1
        elif result[i] == 0: 
            true_negative += 1
        else:
            print('error class not found')
            
#Criação da matriz de confusão
test_result = [false_negative,
               true_negative,
              true_positive,
               false_positive]
adjusted_test_result = np.array(test_result)/(len(test_features))
df_test_result = pd.DataFrame(np.array(['false_negative','true_negative','true_positive','false_positive']),columns=['results'])
df_test_result['percent'] = adjusted_test_result
df_test_result

#Plotagem da matriz de confusão
fig,ax = plt.subplots()
heatmap = ax.pcolor((df_test_result['percent'].to_numpy().reshape(2,2)),cmap='Blues')
data = df_test_result['percent'].to_numpy().reshape(2,2)* 100
for y in range(data.shape[0]):
    for x in range(data.shape[1]):
        ax.text(x + 0.5, y + 0.5, '%.2f' % data[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
ax.set_xticks([0.5,1.5])
ax.set_yticks([0.5,1.5])
ax.set_xticklabels(['Positivo','Negativo'])
ax.set_yticklabels(['Negativo','Positivo'])

plt.xlabel('Valor atual')
plt.ylabel('Valor predito')
plt.title('Matriz de confusão do CNN com %.2f' %(accuracy*100)+ '% de acurácia')
plt.savefig('90_87_model.png')
plt.show()

#Cálculo da precisão, recall e F1 score
Precisao = (true_positive)/(true_positive + false_positive)
recall = (true_positive)/(true_positive + false_negative)
F1Score = (2*Precisao*recall)/(Precisao + recall)

print('A precisão da classificação é: ',Precisao)
print('A recall da classificação é: ',recall)
print('A F1 Score da classificação é: ',F1Score)