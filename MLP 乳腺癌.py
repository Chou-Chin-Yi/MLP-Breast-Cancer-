from sklearn import tree
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

df = pd.read_excel("Breast Cancer.xlsx")

# 判斷資料
train_x_col = ['radius_mean', 'texture_mean',
               'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
               'concavity_mean', 'concave points_mean', 'symmetry_mean',
               'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
               'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
               'concave points_se', 'symmetry_se', 'fractal_dimension_se',
               'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
               'smoothness_worst', 'compactness_worst', 'concavity_worst',
               'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']


# 將資料 轉成陣列
data = df[train_x_col]
data = np.array(data)  # 二微陣列
train_y_col = df["target"]
train_y_col = np.array(train_y_col)  # 一微陣列

print("====標準化==============")
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)
# print("表準化後X的資料為", data)

# 決策數
# 切割80% 訓練資料 和 20% 的測試資料
X_train, X_test,\
y_train, y_test = train_test_split(data, train_y_col, test_size=0.2)

# 將訓練結果 轉成 單熱編碼
y_train2 = tf.keras.utils.to_categorical(y_train, num_classes=2)

# 初始化
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(units=100, activation=tf.nn.relu, input_dim=30),
  tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
  tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),     # 使用Adam 移動0.001
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs")       # 儲存到logs/

# 訓練
history = model.fit(X_train, y_train2,
                    epochs=4000,
                    batch_size=64,
                    callbacks=[tensorboard],
                    verbose=1)

# 將測試結果 轉成 單熱編碼
y_test2 = tf.keras.utils.to_categorical(y_test, num_classes=2)

# 測試分數
score = model.evaluate(X_test, y_test2, batch_size=64)

# 預測答案
predict = model.predict(X_test)
print("predict:", predict[0:10])

ans = ""
for x in range(0, 10):
    ans = ans + str(np.argmax(predict[x])) + " "

print("預測答案(前10筆):", ans)
print("原始答案(前10筆):", y_test[0:10])
print("損失率:", round(score[0], 2), "正確率:", round(score[1], 2))

# 訓練過程圖像
plt.plot(history.history["accuracy"])
plt.plot(history.history["loss"])
plt.title("model accuracy")
plt.ylabel("acc & loss")
plt.xlabel("epoch")
plt.legend(["acc", "loss"], loc="upper right")
plt.show()
