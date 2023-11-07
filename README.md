# football-play-prediction

## Overview

This project aims to develop a deep learning model using Long Short-Term Memory (LSTM) networks to predict football plays based on a series of previous plays. By analyzing the time series data of multiple plays, the model aims to provide accurate predictions, enabling insights into the strategic decisions made by football teams.

##### Table of Contents

* [Dataset](#dataset)  
* [Model Architecture](#architecture)
* [Evaluation](#evaluation)
* [Results](#results)
* [Future Improvements](#improvements)


## Dataset

This data is collected by Daren Willman of NFLsavant.com. For each season a CSV file is provided with each line representing a play.

To use this data in a LSTM Network, plays are stacked in groups of three consecutive plays to predict the type of the following fourth play. This process is done in the script LSTM_modeling.py.

Source: https://nflsavant.com/about.php
Format: CSV

## Model Architecture

The LSTM network in this project is designed to process input sequences consisting of 10 variables. It comprises two LSTM layers followed by a Softmax output layer.

Input Layer:

Input Size: 10 (more features should be added to improve performance)
This layer receives sequences of 10 variables at each time step, with 3 time steps being provided.

LSTM Layer 1:

    Hidden Size: 256
    Number of LSTM units: 256
    Activation Function: Tanh (default activation function for LSTM units)
    The first LSTM layer processes the input sequences and extracts temporal features.

LSTM Layer 2:

    Hidden Size: 256
    Number of LSTM units: 256
    Activation Function: Tanh (default activation function for LSTM units)
    The second LSTM layer further refines the temporal features learned by the first layer.

Output Layer (Softmax):

    Activation Function: Softmax
    The Softmax layer transforms the output of the last LSTM layer into a probability distribution over the two categories RUN and PASS.


## Evaluation

After training the model on the available data multiple standard metrics are calculated. This process evaluates the performance by comparing predicted and true playtype. In order to avoid data leakage a holdout set of plays, not used in the inital training is used.

The metrics used are:
* Accuracy
* Precision
* Recall
* F1-Score
* Matthews Correlation Coefficient 

## Results

This project is currently still ongoing. Experiments are done, but this approach currently yields either massivly overfit models or strongly underfit models. Further improvements to the data and model architecture are needed.

## Future Improvements

To improve the performance of the model mutilple steps are planned.

For example:
* Enhancing existing data through feature engineering
* Making the model independent of drive length
