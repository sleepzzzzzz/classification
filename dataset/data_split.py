import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 数据读取
data = pd.read_csv('hepatoprotection.csv', delimiter=',')

# 数据划分
X = np.array(data['smiles'])
Y = np.array(data['label'])
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2,   # 10%数据作为测试集
                                                stratify=Y,            # 按照标签来分层采样
                                                shuffle=True,          # 是否先打乱数据的顺序再划分
                                                random_state=1024)

# 将训练集转化为csv
train = pd.concat([pd.DataFrame(trainX), pd.DataFrame(trainY)], axis=1)
train.columns = ['smiles', 'label']
train.to_csv('train.csv', index=False, sep=',')
testX, valX, testY, valY = train_test_split(testX, testY, test_size=0.5,
                                            stratify=testY,  # 按照标签来分层采样
                                            shuffle=True,    # 是否先打乱数据的顺序再划分
                                            random_state=1024)

# 将测试集转换为csv
test = pd.concat([pd.DataFrame(testX), pd.DataFrame(testY)], axis=1)
test.columns = ['smiles', 'label']
test.to_csv('test.csv', index=False, sep=',')

# 将验证集转换为csv
validation = pd.concat([pd.DataFrame(valX), pd.DataFrame(valY)], axis=1)
validation.columns = ['smiles', 'label']
validation.to_csv('validation.csv', index=False, sep=',')
