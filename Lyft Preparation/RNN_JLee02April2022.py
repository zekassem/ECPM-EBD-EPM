import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Class example of Temporal learning, Recurrent Neural Network
def preprocessing(data, sequence_length, horizon, stride, val_size, test_size, scaler):
    X = data.values
    Y = X[:, -1].reshape((-1,1)) #assumes the time series to be predicted is the last column
    scaler.fit(X)
    X_norm = scaler.transform(X) # keep the normalized data to predict later
    Y_scaler = scaler.fit(Y)     # keep the scaler attributes to reverse the transform later

    # Sequence Generation Part
    x_seq = []
    y_seq = []

    for i in range(0, len(X_norm) + 1 - sequence_length - horizon, stride):
        x_seq.append(X_norm[i: i + sequence_length])
        y_seq.append(X_norm[i + sequence_length - 1 + horizon, -1])

    x_seq = np.asarray(x_seq)
    y_seq = np.asarray(y_seq)

    # Test data split
    X_train, X_test, Y_train, Y_test = train_test_split(x_seq, y_seq, test_size=test_size, shuffle=False,
                                                        random_state=1004) # random_state ignored because shuffle = False
    # Validation data split
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_size, shuffle=False,
                                                      random_state=1004)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_norm, Y_scaler


def data_loader_generation(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    # train / validation / test sets defined

    train = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    valid = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))
    test = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test))

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


    def forward(self, x): #RNN and GRU code
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        out, hn = self.model(x, h0)
        out = out[:,-1,:]
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
        loss = nn.MSELoss()(output, targets.view(-1,1))
        return loss

    def validation_step(self, batch):
        # Load batch
        features, targets = batch
        # Generate predictions
        output = self(features)
        # Calculate loss
        loss = nn.MSELoss()(output, targets.view(-1,1))
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
            print(f"Epoch {epoch}: \ Train loss: {train_epoch_loss / len(train_loader):.5f} \ Val loss: {val_epoch_loss / len(valid_loader):.5f}")
        return loss_stats["train"], loss_stats["val"]

def final_prediction(data, sequence_length, horizon, batch_size, model, Y_scaler, X_norm ):
    # Prediction results for training and testing
    actual = data.values[:, -1]
    actual = actual[sequence_length + horizon - 1:] # actual value corresponding to first predicted value

    x_seq = []
    for i in range(0, len(X_norm) + 1 - sequence_length - horizon):
        x_seq.append(X_norm[i:i + sequence_length])
    x_seq = np.asarray(x_seq)
    data_loader = torch.utils.data.DataLoader(dataset=torch.Tensor(x_seq), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        y_pred = []
        for data in data_loader:
            out = model(data)
            y_pred += out.cpu().numpy().tolist()

    y_pred = Y_scaler.inverse_transform(y_pred) # reverse scaler to plot later in the original scale
    return actual, y_pred

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

def plot_prediction_graph(actual: object, y_pred: object, test_boundary: object, example: object, titleinfo: object) -> object:

    plt.figure(figsize=(15, 10))
    x = list(range(sequence_length+1, sequence_length+1+len(y_pred)))
    plt.axvline(x=test_boundary, ls= ':', color='black')
    plt.plot(x, actual, 'r--')
    plt.plot(x, y_pred, 'b', linewidth=0.6)
    plt.legend(['test boundary', 'actual', 'prediction'])
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.title('Example '+ example + '  ' + titleinfo)
    plt.show()





# Main function
if __name__ == '__main__':
    # data loading
    # The data used in this example shows 431 days of stock values
    # The data consists Date, Open value, Daily High, Daily low, Close value, Adj close, Trade volume.
    data = pd.read_csv('./stock_value.csv')
    # Data preprocessing parameters
    sequence_length = 10
    horizon = 1
    stride = 1
    test_size = 0.2  # ratio of total sample size, could be defined as an integer
    val_size = 0.2  # ratio of val / (val + train), could be defined as an integer
    scaler = MinMaxScaler() # scaler applied to the predictors


    # model parameters
    hidden_size = 8   # length of hidden sizes
    num_layers = 1
    batch_size = 30
    output_size = 1    # regression
    # training parameters
    epochs = 100
    lr = 0.01
    opt = optim.Adam

    # either 'single' or 'multiple'
    #  'single': use sequence_length of previous values as predictors
    #  'multiple' uses sequence_length of previous values for all predictors (including the target time series)
    example = 'single'  # either 'single' or 'multiple'

    # Model parameter
    if example == 'multiple':
        data = data[['Open', 'Low', 'High', 'Volume', 'Close']]
    else:
        data = pd.DataFrame(data['Close'])

    input_size = len(data.columns)
    # Data preprocessing / scaling : input-> (data, parameter), output->batch loader
    X_train, Y_train, X_val, Y_val, X_test, Y_test, X_norm, Y_scaler = preprocessing(data=data, sequence_length=sequence_length,
                                                                          horizon=horizon, stride=stride,
                                                                          val_size=val_size,
                                                                          test_size=test_size, scaler=scaler)

    # Data_loader generation
    train_loader, valid_loader, test_loader = data_loader_generation(X_train=X_train,
                                                                     Y_train=Y_train, X_val=X_val, Y_val=Y_val,
                                                                     X_test=X_test, Y_test=Y_test)

    # Model generation
    model = Model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, device='cpu')

    # Model trainer generation
    model_trainer = ModelTrainer()

    # Model train
    train_loss, valid_loss = model_trainer.fit(epochs=epochs, learning_rate=lr, model=model, train_loader=train_loader,
                                               valid_loader=valid_loader, opt=opt)
    # Parameter summary to add to output
    parms = {'sequence_length': sequence_length, 'horizon': horizon, 'stride': stride, 'test_size': test_size,
             'val_size': val_size, 'num_layers': num_layers, 'hidden_size': hidden_size, 'batch_size': batch_size}
    titleinfo = ''
    for k, v in parms.items():
        titleinfo += k + " " + str(v) + ', '

    # Plot loss by epochs
    plot_loss_graph(train_loss=train_loss, valid_loss=valid_loss, example = example, titleinfo = titleinfo)

    # Predict all values from trained model
    actual, y_pred = final_prediction(data=data, sequence_length=sequence_length,
                                      horizon=horizon, batch_size=batch_size, model=model, X_norm = X_norm, Y_scaler=Y_scaler)

    # Plot prediction results
    if test_size < 1:
        test_boundary = len(data) - (len(data)*test_size)
    else:
        test_boundary = len(data) - sequence_length - test_size

    plot_prediction_graph(actual=actual, y_pred=y_pred, test_boundary=test_boundary, example = example, titleinfo = titleinfo)

