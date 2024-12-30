import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# Class example of Temporal learning, Recurrent Neural Network

def preprocessing(data, val_size, test_size):
    X = data.values
    Y = X[:, -1].reshape((-1, 1))  # assumes the time series to be predicted is the last column
    X = np.transpose(X[:, :-1])

    scaler.fit(X)
    X_norm = np.transpose(scaler.transform(X))  # keep the normalized data to predict later

    # Y_scaler = scaler.fit(Y)     # keep the scaler attributes to reverse the transform later
    # Test data split

    X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size=test_size, shuffle=True,
                                                        random_state=1004)

    # Validation data split
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_size, shuffle=True,
                                                      random_state=1004)

    # 2D data to 3D for input to RNN
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_norm


def data_loader_generation(X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_size):
    # train / validation / test sets defined
    train = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(Y_train))
    valid = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(Y_val))
    test = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(Y_test))

    # Gererate train / valid / test loader

    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(dataset=valid, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


class Model(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(Model, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Create output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Creating layers with different hidden sizes
        # for i in range(0, num_layers):
        #     self.hidden_layers.append(nn.RNN(self.input_size, self.hidden_sizes[i], batch_first=True))
        #     self.input_size = self.hidden_sizes[i]

    def forward(self, x):  # RNN and GRU code
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        out, hn = self.model(x, h0)
        out = out[:, -1, :]

        # many to one: "out" contains the output from the last hidden layer at EACH time step
        # in the second dimension of the tensor. This code only sends the output from the
        # last time step to the output layer, but all time steps can be used

        out = out.reshape(out.shape[0], -1)
        # Get predictions from the output linear
        out = self.output_layer(out)

        return out

    # def forward(self, x): #LSTM code
    #     h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
    #     # if LSTM, the following is required.
    #     c0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
    #     out, (hn, cn) = self.model(x, (h0, c0))
    #     out = out[:,-1,:]                              # many to one
    #     out = out.reshape(out.shape[0], -1)  # many to many strategy
    #     out = self.output_layer(out)
    #     return out

    def training_step(self, batch):
        # Load batch
        features, targets = batch
        # Generate predictions
        output = self(features)
        # Calculate loss
        targets = targets.reshape(targets.shape[0])
        loss = nn.CrossEntropyLoss()(output, targets)

        return loss

    def validation_step(self, batch):
        # Load batch
        features, targets = batch
        # Generate predictions
        output = self(features)
        targets = targets.reshape(targets.shape[0])
        # Calculate loss
        loss = nn.CrossEntropyLoss()(output, targets)

        return loss


class ModelTrainer():

    def fit(self, epochs, learning_rate, model, train_loader, valid_loader, opt):
        loss_stats = {"train": [], "val": []}
        optimizer = opt(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            # Training
            train_epoch_loss = 0

            for batch in train_loader:
                loss = model.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_epoch_loss += loss.item()

            # Validation
            with torch.no_grad():
                val_epoch_loss = 0
                for batch in valid_loader:
                    loss = model.validation_step(batch)
                    val_epoch_loss += loss.item()

            loss_stats["train"].append(train_epoch_loss / len(train_loader))
            loss_stats["val"].append(val_epoch_loss / len(valid_loader))

            print(
                f"Epoch {epoch}: \ Train loss: {train_epoch_loss / len(train_loader):.5f} \ Val loss: {val_epoch_loss / len(valid_loader):.5f}")

        return loss_stats["train"], loss_stats["val"]

def final_prediction_classifier(test_loader, model):
    # X_test = torch.utils.data.TensorDataset(X_test, Y_test)
    # test_loader = torch.utils.data.DataLoader(dataset=X_test, batch_size=batch_size, shuffle=False)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch in test_loader:
            x_test, y_test = batch
            output = model(x_test)
            output = torch.argmax(output, 1).numpy()
            y_pred.extend(output)  # Save Prediction

            y_test = y_test.numpy().reshape(y_test.shape[0])
            y_true.extend(y_test)  # Save Truth

    # Accuracy & Confusion Matrix
    print("accuarcy_score=",accuracy_score(y_true, y_pred))
    cf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cmtx = pd.DataFrame(cf_matrix,index=['true:0', 'true:1'], columns=['pred:0', 'pred:1'])
    print("confusion_Matrix", cmtx)

    return cmtx

def plot_loss_graph(train_loss, valid_loss, example, titleinfo):
    plt.figure(figsize=(15, 10))
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.xlabel('epochs')
    plt.ylabel('AVG_running_MSE_loss')
    plt.legend(['train', 'validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Close')
    plt.title('Example ' + example + '  ' + titleinfo)
    plt.show()


# Main function

if __name__ == '__main__':

    data = pd.read_excel('./acttimeseriesmultipleinstances.xlsx')

    # Data preprocessing parameters

    test_size = 0.3  # ratio of total sample size, could be defined as an integer
    val_size = 0.2  # ratio of val / (val + train), could be defined as an integer
    scaler = MinMaxScaler()  # scaler applied to the predictors

    # model parameters

    hidden_size = 8  # number of hidden nodes in each hidden layer
    num_layers = 1
    batch_size = 2
    output_size = 2  # 0-1 Classification problem

    # training parameters

    epochs = 50
    lr = 0.03

    opt = optim.Adam
    example = 'classify'

    # data = data[['Open', 'Low', 'High', 'Volume', 'Close']]
    sequence_length = len(data.columns)-1  # All columns but last column.

    # Model parameter
    input_size = 1        #len(data.columns)-1


    # Data preprocessing / scaling : input-> (data, parameter), output->batch loader
    X_train, Y_train, X_val, Y_val, X_test, Y_test, X_norm = preprocessing(data=data,val_size=val_size,
                                                                           test_size=test_size)

    # Data_loader generation
    train_loader, valid_loader, test_loader = data_loader_generation(X_train=X_train,
                                                                     Y_train=Y_train, X_val=X_val, Y_val=Y_val,
                                                                     X_test=X_test, Y_test=Y_test,
                                                                     batch_size=batch_size)

    # Model generation
    model = Model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size,
                  device='cpu')

    # Model trainer generation
    model_trainer = ModelTrainer()

    # Model train
    train_loss, valid_loss = model_trainer.fit(epochs=epochs, learning_rate=lr, model=model, train_loader=train_loader,
                                               valid_loader=valid_loader, opt=opt)

    # Parameter summary to add to output
    parms = {'sequence_length': sequence_length, 'test_size': test_size,
             'val_size': val_size, 'num_layers': num_layers, 'hidden_size': hidden_size, 'batch_size': batch_size}

    titleinfo = ''
    for k, v in parms.items():
        titleinfo += k + " " + str(v) + ', '

    # Plot loss by epochs
    plot_loss_graph(train_loss=train_loss, valid_loss=valid_loss, example=example, titleinfo=titleinfo)

    # Accuarcy and Confusion Matrix for test data
    final_prediction_classifier(test_loader, model)

