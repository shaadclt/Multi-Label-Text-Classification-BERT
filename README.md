# Multi-Label Text Classification with BERT

This project focuses on multi-label text classification using BERT (Bidirectional Encoder Representations from Transformers). We combine titles and abstracts of articles to classify them into multiple categories simultaneously.

## Dataset

The dataset used in this project includes titles and abstracts of articles along with their associated categories. Categories with low frequencies were dropped to focus on the significant ones.

## Requirements

Ensure you have the following libraries installed:

- pandas
- matplotlib
- numpy
- torch
- transformers
- scikit-learn

## Data Preparation

### Load and Preview the Dataset
First, the dataset is loaded and a preview of the first few rows is displayed to understand its structure.

### Data Cleaning and Preparation
A bar chart is plotted to show the frequency of each category. Categories with low frequencies are dropped to ensure the model focuses on significant ones. Articles that do not belong to any of the remaining categories are also removed.

### Combine Title and Abstract
The title and abstract of each article are combined into a single string to increase the amount of information available for classification. The original title and abstract columns are then dropped.

### Split the Dataset
The dataset is split into training, validation, and test sets to allow for proper training and evaluation of the model.

## Model Training

### Hyperparameters
Several hyperparameters are defined, including the maximum length of token sequences, batch sizes for training, validation, and testing, the number of epochs, learning rate, and the threshold for classification.

### Tokenizer Initialization
The BERT tokenizer is initialized to convert text into tokens that can be fed into the BERT model.

### Custom Dataset Class
A custom dataset class is created to handle the encoding of text and the preparation of input data for the BERT model. This class ensures that each text is properly tokenized and padded/truncated to the specified maximum length.

### Model Definition
A custom BERT-based model class is defined. This class includes a BERT model with a dropout layer and a linear layer for classification. The forward method specifies how the input data passes through the model to produce output predictions.

### Training and Evaluation Functions
Functions are defined for training and evaluating the model. The training function handles the forward and backward passes, gradient clipping, and updating of model parameters. The evaluation function assesses the model's performance on the validation set without updating the model parameters.

### Model Training Loop
A training loop iterates over a specified number of epochs. In each epoch, the model is trained on the training set and evaluated on the validation set. The accuracy and loss for both training and validation are tracked. The best model (based on validation accuracy) is saved.

## Model Evaluation

### Load the Best Model
The best model, saved during training, is loaded for evaluation on the test set.

### Evaluate the Model
The model's accuracy and loss are calculated on the test set to assess its performance.

### Generate Classification Report
A classification report is generated using the test set to provide detailed metrics, such as precision, recall, and F1-score, for each category.

### Test on New Data
The model is tested on new, unseen data to demonstrate its ability to classify new articles. The text is tokenized, fed into the model, and the predicted categories are displayed.

## Conclusion

This project demonstrates how to perform multi-label text classification using BERT. The model was trained and evaluated using a dataset of article titles and abstracts, and it achieved significant accuracy in classifying the articles into multiple categories.
