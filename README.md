# Neural Machine Translation from Afrikaans to English 

This repository contains the implementation of Neural Machine Translation from Afrikaans to English.

### Dataset

The dataset contains pairs of English - Afrikaans sentences. ```Afr.txt```  was gotten from [Tatoeba Project](https://tatoeba.org/).

### Preprocessing the dataset

```Afrikaans_English.ipynb```: The data was cleaned and broken into smaller training and testing dataset. Then ```english-afrikaan-both.pkl```, ```english-afrikaan-train.pkl``` and ```english-afrikaan-test.pkl```  were generated for training and testing purposes.

The preprocessing of the data involves:

- Removing punctuation marks from the data.

- Removing  all non-printable characters.

- Normalizing all Unicode characters to ASCII (e.g. Latin characters).

- Converting text corpus into lower case characters.

- Shuffling the sentences as sentences were previously sorted in the increasing order of their length.

- Training the Encoder-Decoder LSTM model

After  training, the model will be saved as ```model.h5``` in your directory.

This model uses Encoder-Decoder LSTMs for NMT. In this architecture, the input sequence is encoded by the front-end model called encoder then, decoded by backend model called decoder.

It uses **Adam Optimizer** to train the model using Stochastic Gradient Descent and minimizes the categorical loss function.

### Evaluating the model

Run evaluate_model.py to evaluate the accuracy of the model on both train and test dataset.

- It loads the best saved ```model.h5``` model.

- The model performs pretty well on train set and have been generalized to perform well on test set.


### Reports

The report on this project can be found [here](https://drive.google.com/file/d/1ZW9OVuWEo9QNbF5Z8lFXP6HO21YGDQsW/view?usp=drivesdk)


### References

This work builds extensively on the following works:

1. G. Lample, A. Conneau, L. Denoyer, MA. Ranzato, Unsupervised Machine Translation With Monolingual Data Only, 2018a. (https://arxiv.org/abs/1711.00043)

Thanks to [Tatoeba Project](https://tatoeba.org/) for the dataset.


## License

See the [LICENSE](https://github.com/makozi/Udacity-ML-Engineer-Capstone-Project/blob/master/LICENSE) file for more details.


##### By Marizu-Ibewiro Makozi.
