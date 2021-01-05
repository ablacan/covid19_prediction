# covid19_prediction
Repo of covid19 prediction Models and Data Visualization.

This work is related to the XPrize Pandemic Response Challenge. The challenge's first phase goal was to develop a model able to predict new cases of Covid19 over 182 countries and over several time horizons. A lot of work regarding the code and the modelling has been made in collaboration with my team 'Transatlantic team' participating in the XPrize Pandemic Response Challenge.

Requirements :\
Install `pandas`, `numpy`, `matplotlib`, `seaborn`, `Trendy`, `scikit-learn`.

Scripts:\
In both scripts, functions `PreProcessing` will return the preprocessed dataset. Also, functions `GetCovidPredictions` will perform the preprocessing, training and prediction according to the specified arguments. It returns both the predictions dataframe and the list of each country's MAE.

BASELINE \
Import the `COVID_Predictor_baseline` class function from `script_model_predictions_baseline.py`. Call `GetCovidPredictions` function and specify whether to predict on 4 days, 7 days or 30 days (`period`), the number of lookback days to take into account `nb_lookback_days`, whether to train on data up to January 2020 or only up to August 2020 (`reduced_train`=True) and whether or not to train and predict using different lookback days (`test_several_lookbacks`, returns only the global MAE).

MODEL WITH ADDITIONAL GOOGLE MOBILITY DATA \
Import the `COVID_Predictor` class function from `script_model_predictions_baseline.py` by specifying 'RIDGE', otherwise the class's default model is Lasso. 
Call `GetCovidPredictions` function and specify whether to predict on 4 days, 7 days or 30 days (`period`), the number of lookback days to take into account `nb_lookback_days`, whether to train on data up to January 2020 or only up to August 2020 (`reduced_train`=True) and whether or not to train and predict using different lookback days (`test_several_lookbacks`, returns only the global MAE).

VISUALIZATION\
Data visualization is presented in the `Data_Visualization` jupyter notebook. Model results are presented in the jupyter notebooks `Model_results` and `Model_results_reduced_trainingset` (where models are trained on data only up to August 2020).
