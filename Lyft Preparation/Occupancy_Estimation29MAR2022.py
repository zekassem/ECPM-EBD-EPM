import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Data Reading from existing csv file
df = pd.read_csv('./Occupancy_Estimation.csv')
pd.set_option('display.max_columns', None)
print("data_before_scaling", df)


# Data Scaling for training data
scaler = MinMaxScaler()
# df[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound',
#      'S3_Sound', 'S4_Sound', 'S5_CO2', 'S5_CO2_Slope']] \
#      = scaler.fit_transform(df[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound',
#      'S3_Sound', 'S4_Sound', 'S5_CO2', 'S5_CO2_Slope']])
df[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']] = scaler.fit_transform(df[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']])

# X = df[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound',
#     'S3_Sound', 'S4_Sound', 'S5_CO2', 'S5_CO2_Slope','S6_PIR','S7_PIR']].values
X = df[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']].values
y = df['Room_Occupancy_Count'].values
test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,  shuffle=True, random_state=1004)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
test_size = y_test.shape[0]

features_train_tensor = torch.tensor(X_train)
target_train_tensor = torch.tensor(y_train)
train = torch.utils.data.TensorDataset(features_train_tensor, target_train_tensor)

batch_size = 30
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class NN_STRUCTURE(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.fc1 = nn.Linear(input, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, output, bias=True)
    
    def forward(self, x):
        #print(x)
        x = F.leaky_relu_(self.fc1(x))
        x = F.leaky_relu_(self.fc2(x))
        return x

input_size = X_train.shape[1]
hidden_size = 8
output_size = 4
model_NN = NN_STRUCTURE(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss().to(device)            # Includes softmax
optimizer = optim.Adam(model_NN.parameters(), lr=0.05)
num_epochs = 100

# containers for stats
loss_stats = {   "train": []   }

for epoch in range(num_epochs):
    train_epoch_loss = 0
    total_batch = len(train_loader)

    for X_train_batch, y_train_batch in train_loader:
        X_train_batch = X_train_batch.to(device)
        Y_train_batch = y_train_batch.to(device)

        optimizer.zero_grad()
        y_train_pred = model_NN(X_train_batch)
        train_loss = criterion(y_train_pred, y_train_batch)
        train_loss.backward()
        optimizer.step()
        train_epoch_loss += train_loss.item()

    loss_stats["train"].append(train_epoch_loss / len(train_loader))
    print(f"Epoch {epoch}: \ Train loss: {train_epoch_loss / len(train_loader):.5f}")
plt.figure(figsize=(15, 10))
plt.plot(loss_stats["train"])
plt.xlabel('epochs')
plt.ylabel('AVG_running_loss')
plt.legend(['train'])
plt.show()

features_test_tensor = torch.tensor(X_test)
target_test_tensor = torch.tensor(y_test)

test = torch.utils.data.TensorDataset(features_test_tensor, target_test_tensor)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    sum =0
    for data in test_loader:
        X_test, y_test = data
        y_hat = model_NN(X_test)
        correct_prediction = torch.argmax(y_hat, 1) == y_test
        sum = sum + torch.sum(correct_prediction.float())
    accuracy = sum / torch.tensor(test_size)
    print('Accuracy:', accuracy.item())



