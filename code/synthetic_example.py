import numpy as np

import torch
import torch.nn as nn

from itertools import chain, combinations

import utils.models as models


### DATA ###
def generate_signal(freq, signal_length=128, uniform_amplitude=True, noise_strength=0.01, random_phase=True):
    if random_phase:
        phase_shift = np.random.rand(len(freq)) * 2*np.pi
    else:
        phase_shift = np.zeros(len(freq))

    if uniform_amplitude:
        amplitude = np.ones(len(freq), dtype=np.float)
    else:
        amplitude = np.random.rand(len(freq)) +0.2
    
    x = np.linspace(0,1, signal_length)
    signal = np.zeros_like(x)
    for f,p,a in zip(freq, phase_shift, amplitude):
        signal += a* np.sin(2*np.pi*f*x + p)

    # add noise
    noise = np.random.normal(size=signal_length)
    signal += noise_strength * noise

    return signal


def generate_signal_data(signal_length = 512,
                        n = 10000,
                        n_freq_min=5,
                        n_freq_max=10,
                        freq_interval=1,
                        freq_choice=None,
                        random_n_freq=True,
                        max_freq=10,
                        noise_strength=0.001,
                        random_phase=True,
                        uniform_amplitude=True,
                        weighted_sampling=True,
                        freqs_for_powerset_label=None,
                        weight=7
                        ): 

    if freq_choice is not None:
        weights = np.ones_like(freq_choice)
    else:
        weights = np.ones(max_freq-1)
        if weighted_sampling:
            weights[np.array(freqs_for_powerset_label)-1] = weight
    weights = weights / weights.sum()

    if freq_choice is None:
        freq_choice = np.arange(1,max_freq,freq_interval)
    else:
        n_freq_min, n_freq_max = 0,len(freq_choice)

    x = np.zeros((n, signal_length))
    y = []
    for i in range(n):
        if random_n_freq:
            n_freq = np.random.choice(np.arange(n_freq_min,n_freq_max), 1)
        else:
            n_freq = n_freq_max
        freq = np.random.choice(freq_choice, replace=False, size=n_freq, p=weights)
        x[i] = generate_signal(freq,signal_length=signal_length, noise_strength=noise_strength, random_phase=random_phase, uniform_amplitude=uniform_amplitude)
        y.append(freq)

    if freqs_for_powerset_label is not None:
        freq_combinations = [np.array(a) for a in powerset_sub(freqs_for_powerset_label, None)]
        label = label_f_stationary(y, freq_combinations)
    else:
        label = None

    return x,y,label
def powerset_sub(iterable, length):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    if length is None:
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    else:
        return list(combinations(s, length))


def label_f_stationary(y, freq_combinations):
    """"
    y: list of frequencies in sample
    freq_combinations: list of lists with those frequency combinations {k*}_i that receive a label
    """
    label = np.zeros(len(y))
    for i in range(len(y)):
        y_i = y[i]

        for i_c,c in enumerate(freq_combinations):
            if np.isin(c, y_i).all():
                label[i] = i_c
    return label.astype(int)

### Model ###

def train_model(x_train, y_train, signal_length, cnn, n_layer, n_label, batch_size, lr, epochs, cuda, binary=True):
    #model = models.MultiLabelClassificationModel(signal_length=signal_length, n_layer=n_layer, n_label=n_label)
    if cnn:
        model = models.CNN_1d( n_out=n_label)
    else:
        model = models.MLPModel(signal_length=signal_length, n_out=n_label, n_layer=n_layer)
    
    if cuda:
        model = model.cuda()
        
    data_tensor = torch.tensor(x_train, dtype=torch.float32)
    if cnn:
        data_tensor = data_tensor.unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(data_tensor, torch.tensor(y_train, dtype=torch.long))
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    if binary:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):  
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:    # print every 2000 mini-batches
                print('epoch: %d, loss: %.3f' %
                    (epoch + 1, running_loss / 50))
                running_loss = 0.0

    print('Finished Training')

    model = model.eval()

    return model
