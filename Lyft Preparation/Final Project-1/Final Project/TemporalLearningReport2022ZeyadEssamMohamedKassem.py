import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import csv


# Class example of Temporal learning, Recurrent Neural Network
def preprocessing(data, sequence_length, horizon, stride, val_size, scaler):
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
        y_seq.append(X_norm[i + sequence_length:i + sequence_length+horizon,-1])


    x_seq = np.asarray(x_seq)
    y_seq = np.asarray(y_seq)


    # Validation data split
    X_train, X_val, Y_train, Y_val = train_test_split(x_seq, y_seq, test_size=val_size, shuffle=False,
                                                      random_state=1004)

    return X_train, Y_train, X_val, Y_val, X_norm, Y_scaler


def data_loader_generation(X_train, Y_train, X_val, Y_val):
    # train / validation / test sets defined

    train = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    valid = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))

    # Gererate train / valid / test loader
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(dataset=valid, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(Model, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = nn.RNN(input_size, hidden_size, num_layers, batch_first=True) # what is the batch first?
        # Create output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Creating layers with different hidden sizes
        # for i in range(0, num_layers):
        #     self.hidden_layers.append(nn.RNN(self.input_size, self.hidden_sizes[i], batch_first=True))
        #     self.input_size = self.hidden_sizes[i]


    def forward(self, x): #RNN and GRU code
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device) # Don't understand this part
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
        loss = nn.L1Loss()(output, targets)
        return loss

    def validation_step(self, batch):
        # Load batch
        features, targets = batch
        # Generate predictions
        output = self(features)
        # Calculate loss
        loss = nn.L1Loss()(output, targets)
        return loss

class ModelTrainer():
    def fit(self, epochs, learning_rate, model, train_loader, valid_loader, opt):
        loss_stats = {"train": [], "val": []}
        optimizer = opt(model.parameters(), lr=learning_rate)
        with open('Error_seq_length_' + str(sequence_length) + '_num_layers_' + str(num_layers) + '_hidden_size_' + str(hidden_size) + '_batch_size' + str(batch_size)  + '.csv', 'w') as newFile:
            newFileWriter = csv.writer(newFile, lineterminator='\n')
            newFileWriter.writerow(['Epoch', 'Train loss', 'Val loss'])
            for epoch in range(epochs):
                # Training
                train_epoch_loss = 0
                for batch in train_loader:
                    loss = model.training_step(batch)
                    # calculating gradients
                    loss.backward()
                    # Updating the parameters
                    optimizer.step()
                    # clear gradients
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

                newFileWriter.writerow([epoch, train_epoch_loss / len(train_loader), val_epoch_loss / len(valid_loader)])

        return loss_stats["train"], loss_stats["val"]

def final_prediction(data, sequence_length, horizon, batch_size, model, Y_scaler, X_norm,stride):
    # Prediction results for training and testing
    x_seq = []
    x_seq.append(X_norm[-sequence_length:])
    x_seq = np.asarray(x_seq)
    data_loader = torch.utils.data.DataLoader(dataset=torch.Tensor(x_seq), batch_size=batch_size, shuffle=False)
    with torch.no_grad(): # I think here the model is working without learning
        y_pred = []
        for data in data_loader:
            out = model(data)
            y_pred.extend(out.cpu().numpy().tolist())
    y_pred=np.asarray(y_pred)
    y_pred = Y_scaler.inverse_transform(y_pred)  # reverse scaler to plot later in the original scale
    return y_pred

def plot_loss_graph(train_loss, valid_loss, example, titleinfo):

    plt.figure(figsize=(15, 10))
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.xlabel('epochs')
    plt.ylabel('AVG_running_MAE_loss')
    plt.legend(['train', 'validation'])
    plt.xlabel('Epoch')
    plt.title('Example ' + example + '  ' + titleinfo)
    plt.savefig('Fig_seq_length_' + str(sequence_length) + '_num_layers_' + str(num_layers) + '_hidden_size_' + str(
                hidden_size) + '_batch_size' + str(batch_size)+'_design_1'+'_.png')
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
    data = pd.read_csv('./MTLprojectdata2022.csv')
    data=data.loc[data['location']==6]
    data=data.iloc[::-1] # getting the earlier dates first
    # Data preprocessing parameters
    sequence_length = 20
    horizon = 20
    stride = 1
    val_size = 0.15  # ratio of val / (val + train), could be defined as an integer
    scaler = MinMaxScaler() # scaler applied to the predictors


    # model parameters
    hidden_size = 8 # length of hidden sizes
    num_layers = 2
    batch_size = 120
    output_size = 20    # regression
    # training parameters
    epochs = 200
    lr = 0.01
    opt = optim.Adam

    # either 'single' or 'multiple'
    #  'single': use sequence_length of previous values as predictors
    #  'multiple' uses sequence_length of previous values for all predictors (including the target time series)
    example = 'multiple'  # either 'single' or 'multiple'

    # Model parameter
    if example == 'multiple':
        data = data[[ 'x1','x3','x4', 'x5', 'x6']]
    else:
        data = pd.DataFrame(data[['x6']])

    input_size = len(data.columns)

    # Data preprocessing / scaling : input-> (data, parameter), output->batch loader
    X_train, Y_train, X_val, Y_val, X_norm, Y_scaler = preprocessing(data=data, sequence_length=sequence_length,
                                                                          horizon=horizon, stride=stride,
                                                                          val_size=val_size,
                                                                           scaler=scaler)


    # Data_loader generation
    train_loader, valid_loader = data_loader_generation(X_train=X_train,
                                                                     Y_train=Y_train, X_val=X_val, Y_val=Y_val,
                                                                     )


    # Model generation
    model = Model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, device='cpu')
    # input_size, hidden_size, num_layers, output_size, device

    # Model trainer generation
    model_trainer = ModelTrainer()

    # Model train
    train_loss, valid_loss = model_trainer.fit(epochs=epochs, learning_rate=lr, model=model, train_loader=train_loader,
                                               valid_loader=valid_loader, opt=opt)


    # Parameter summary to add to output
    parms = {'sequence_length': sequence_length, 'horizon': horizon, 'stride': stride,
             'val_size': val_size, 'num_layers': num_layers, 'hidden_size': hidden_size, 'batch_size': batch_size}
    titleinfo = ''
    for k, v in parms.items():
        titleinfo += k + " " + str(v) + ', '

    # Plot loss by epochs
    plot_loss_graph(train_loss=train_loss, valid_loss=valid_loss, example = example, titleinfo = titleinfo)


    # Predict all values from trained model
    y_pred = final_prediction(data=data, sequence_length=sequence_length,
                                      horizon=horizon, batch_size=batch_size, model=model, X_norm = X_norm, Y_scaler=Y_scaler,stride=stride)

    with open('TemporalLearningReport2022ZeyadEssamMohamedKassem.csv','w') as newFile:
        newFileWriter = csv.writer(newFile, lineterminator='\n')
        for i in range(y_pred.shape[1]):
            newFileWriter.writerow(y_pred[:,i])


