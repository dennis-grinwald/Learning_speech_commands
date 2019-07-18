## A convolutional recurrent neural network(CRNN) for source location mapping of spoken words
We provide an end-to-end solution of a convolutional recurrent neural network, that maps sound sequences of spoken words to the corresponding written word labels. 
Having the trained network, we extract the activations of the neurons in the single layers, to map them to source locations in the human brain. 

# Getting started
In order to get started, download the data from https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data and place it into:

```
data/
```

## Best Model Performances
So far the best model has the following performance:

- Best Test Accuracy so far: 74% (accuracy of predicting out of the 31 different class-labels when inputting prior unseen data)
- Best model saved in: trained_models/128_hidden/run2/best_model_1fc.pt
- train/test curves of best model saved in: results/128_hidden/1fc


IMPORTANT: Make sure, that you have the data downloaded and it is residing in the data/ in the root directory. It should be structured as follows: data/marvin/wav-file-1 etc. Though you can easily adjust the data paths in the "training.py" or "simulate_activity.py" scripts if needed want.

## Training
We organized the training procedure of the network as easy as possible. In order to get it running the user just needs to adjust the hyperparameters in the "training.json" file.
These include batch size, hidden neurons, kernel size etc.
However we provide an already pretrained model. Just make sure to have the data in the root directory of this project as described above.

## Simulating source activity

In order to simulate a forward pass of the trained model adjust "model_path" in: "simulate_activity.py" to the most recent model (no need to change path if you use model from the repo, change only if using your own one).

In order to visualize training/testing process of the model adjust "train_curve_path" and "test_curve_path" in: "plot_learning_curves.py" (no need to change paths if you use model from the repo, change only if using your own one).


In order to get the single layer activations run the ”simulate_activity.py” script. Make sure to adjust the “model_path” variable to the most recent model (see above).

The returned matrices in the model.out(data) method return 4 objects. The first 3 correspond to STG, TP, IFG activation matrices for each data input respectively.

The shapes are:

- STG: (batch_size x time_steps x neurons)
- TP: (batch_size x time_steps x neurons)
- IFG: (batch_size x time_steps x neurons)
