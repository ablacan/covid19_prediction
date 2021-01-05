import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore")


class COVID_Predictor():
    """"""
    def __init__(self, RIDGE=False):
        #Url to download Oxford data
        self.DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
        #Variables to keep from Oxford dataset
        self.IP_COLUMNS = ['C1_School closing',
                      'C2_Workplace closing',
                      'C3_Cancel public events',
                      'C4_Restrictions on gatherings',
                      'C5_Close public transport',
                      'C6_Stay at home requirements',
                      'C7_Restrictions on internal movement',
                      'C8_International travel controls',
                      'H1_Public information campaigns',
                      'H2_Testing policy',
                      'H3_Contact tracing',
                      'H6_Facial Coverings']
        self.ID_COLUMNS = ['CountryName','RegionName','GeoID','Date']

        self.CASES_COLUMN = ['NewConfirmedCases_7davg']

        self.MOBILITY_COLUMNS=['retail_and_recreation_percent_change_from_baseline',
        'grocery_and_pharmacy_percent_change_from_baseline',
        'parks_percent_change_from_baseline',
        'transit_stations_percent_change_from_baseline',
        'workplaces_percent_change_from_baseline',
        'residential_percent_change_from_baseline']

        # Training on the full dataset or a subset depending on the train starting date
        self.train_start_date = np.datetime64('2020-07-29')
        self.train_end_date = np.datetime64('2020-11-29')
        # For testing, restrict training data to that before a hypothetical predictor submission date
        self.test_end_date = np.datetime64('2020-12-29')

        if RIDGE:
            self.model= Ridge(alpha=1.0,
                            fit_intercept=True, normalize=False,tol=0.001,
                            solver='auto', random_state=None)
        else:
            # Set positive=True to enforce assumption that cases are positively correlated
            # with future cases and npis are negatively correlated.
            self.model = Lasso(alpha=0.1,
                            precompute=True,
                            max_iter=10000,
                            positive=True,
                            selection='random')

    def PreProcessing(self):
        """"""
        #Get dataset
        df = pd.read_csv(self.DATA_URL,
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str},
                 error_bad_lines=False)

        KEEP_COLUMNS = ["CountryName", "CountryCode", "RegionName", "Date", "Jurisdiction", "ConfirmedCases"]

        #Drop Brazil regions
        df.drop(labels = df[df.CountryName=='Brazil'].index[0:-1], axis=0, inplace=True)
        df = df.reset_index(0, drop=True)

        #Keep only the needed NPIs
        base_df = df[KEEP_COLUMNS + self.IP_COLUMNS]

        # Add GeoID column that combines CountryName and RegionName for easier manipulation of data
        base_df["GeoID"] = np.where(base_df["RegionName"].isnull(),
                                 base_df["CountryName"],
                                 base_df["CountryName"] + ' / ' + base_df["RegionName"])

        #Create DailyChange cases column
        base_df["DailyChangeConfirmedCases"] = base_df.groupby(["GeoID"]).ConfirmedCases.diff().fillna(0)

        #Set to zero negative values (inconsistencies)
        neg_values = np.where(base_df['DailyChangeConfirmedCases'] < 0)
        for i in neg_values:
            base_df['DailyChangeConfirmedCases'].loc[i] = 0

        # Compute the 7 days moving average for PredictedDailyNewCases
        base_df["NewConfirmedCases_7davg"] = base_df.groupby("GeoID")['DailyChangeConfirmedCases'].rolling(7, center=False).mean().reset_index(0, drop=True).fillna(0)

        # Fill any missing NPIs by assuming they are the same as previous day
        for npi in self.IP_COLUMNS:
            base_df.update(base_df.groupby('GeoID')[npi].ffill().fillna(0))

        #Drop non useful columns
        base_df = base_df.drop(['DailyChangeConfirmedCases','ConfirmedCases'], axis=1)


        """Merge Oxford data with Google mobility dataset"""
        #Download mobility dataset
        mobility_df = pd.read_csv('Global_Mobility_Report.csv', parse_dates=['date'])

        #Drop non useful information
        mobility_df.drop(['sub_region_1','sub_region_2', 'metro_area', 'iso_3166_2_code','census_fips_code'], axis=1, inplace=True)

        #Rename column of countries
        mobility_df = mobility_df.rename(columns={'country_region': 'CountryName', 'date':'Date'})

        #Drop duplicates in dates
        mobility_df = mobility_df.drop_duplicates(subset=['CountryName','Date'], keep='first', ignore_index=True)

        #Get merged dataset with NPIs and Mobility Data
        base_df = base_df.merge(mobility_df[self.MOBILITY_COLUMNS + ['CountryName','Date']], left_on=['CountryName','Date'], right_on=['CountryName','Date'], how='left')

        # Fill any missing data by assuming they are the same as previous day + put NaNs to 0 for countries without mobility data
        for mob in self.MOBILITY_COLUMNS:
            base_df.update(base_df.groupby('GeoID')[mob].ffill().fillna(0))

        return base_df


    @staticmethod
    def compute_mae(pred, true):
        """Compute Mean Average Error between predictions and groundtruth"""
        return np.mean(np.abs(pred - true))


    def Training(self, data, verbose=False, nb_lookback_days=20):
        """
            Train the model

            :param data: Training data
            :type data: pandas.DataFrame
            :param verbose: Whether to show traces for debug (True) or run in quiet mode (False)
            :type verbose: bool
        """

        # Create training data across all countries for predicting one day ahead
        x_samples = []
        y_samples = []
        geo_ids = data.GeoID.unique()
        for g in geo_ids:
            gdf = data[data.GeoID == g]
            all_case_data = np.array(gdf[self.CASES_COLUMN])
            all_npi_data = np.array(gdf[self.IP_COLUMNS])
            all_mobility_data = np.array(gdf[self.MOBILITY_COLUMNS])

            # Create one sample for each day where we have enough data
            # Each sample consists of cases and npis for previous nb_lookback_days
            nb_total_days = len(gdf)
            for d in range(nb_lookback_days, nb_total_days - 1):
                x_cases = all_case_data[d - nb_lookback_days:d]

                x_mobility = all_mobility_data[d - nb_lookback_days:d]

                # Take negative of npis to support positive
                # weight constraint in Lasso.
                x_npis = -all_npi_data[d - nb_lookback_days:d]

                # Flatten all input data so it fits Lasso input format.
                x_sample = np.concatenate([x_cases.flatten(),
                                           x_npis.flatten(),
                                           x_mobility.flatten()])
                y_sample = all_case_data[d + 1]
                x_samples.append(x_sample)
                y_samples.append(y_sample)

        x_samples = np.array(x_samples)
        y_samples = np.array(y_samples).flatten()

        # Split data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x_samples,
                                                            y_samples,
                                                            test_size=0.2,
                                                            random_state=301,
                                                           shuffle=False)

        # Fit model
        self.model.fit(x_train, y_train)

        # Evaluate model
        train_preds = self.model.predict(x_train)
        train_preds = np.maximum(train_preds, 0)  # Don't predict negative cases

        test_preds = self.model.predict(x_test)
        test_preds = np.maximum(test_preds, 0)  # Don't predict negative cases
        if verbose:
            print('Test MAE:', self.compute_mae(test_preds, y_test))

        # Inspect the learned feature coefficients for the model
        # to see what features it's paying attention to.

        # Give names to the features
        x_col_names = []
        for d in range(-nb_lookback_days, 0):
            x_col_names.append('Day ' + str(d) + ' ' + self.CASES_COLUMN[0])
        for d in range(-nb_lookback_days, 1):
            for col_name in self.IP_COLUMNS:
                x_col_names.append('Day ' + str(d) + ' ' + col_name)
        for d in range(-nb_lookback_days, 1):
            for col_name in self.MOBILITY_COLUMNS:
                x_col_names.append('Day ' + str(d) + ' ' + col_name)

        # View non-zero coefficients
        for (col, coeff) in zip(x_col_names, list(self.model.coef_)):
            if coeff != 0.:
                print(col, coeff)
        if verbose:
            print('Intercept', self.model.intercept_)



    def Predict(self, start_date, end_date, training_set, test_set, NB_LOOKBACK_DAYS=20, verbose=False):
        """
            Predict new cases between two dates

            :param start_date: First date of the interval in the format (yyyy-mm-dd)
            :type start_date: str
            :param end_date: Last date of the interval in the format (yyyy-mm-dd)
            :type end_date: str
            :param ip_file: Path to the Intervention Plan file
            :type ip_file: str
            :param output_file: File to write the output predictions
            :type output_file: str
            :param verbose: Whether to show traces for debug (True) or run in quiet mode (False)
            :type verbose: bool
        """

        #Set our testing period dates
        start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date, format='%Y-%m-%d')

        #Get historical variables from training set
        hist_variables_df = training_set[self.ID_COLUMNS + self.IP_COLUMNS+ self.MOBILITY_COLUMNS]
        #Get variables from testing set period

        test_set = test_set[(test_set.Date >= start_date) & (test_set.Date <= end_date)]
        variables_df = test_set[self.ID_COLUMNS + self.IP_COLUMNS + self.MOBILITY_COLUMNS]

        # Keep only the id and cases columns
        hist_cases_df = training_set[self.ID_COLUMNS + self.CASES_COLUMN]

        # Make predictions for each country,region pair
        geo_pred_dfs = []
        for g in variables_df.GeoID.unique():
            if verbose:
                print('\nPredicting for', g)

            # Pull out all relevant data for country c
            hist_cases_gdf = hist_cases_df[hist_cases_df.GeoID == g]
            last_known_date = hist_cases_gdf.Date.max()
            vars_gdf = variables_df[variables_df.GeoID == g]

            past_cases = np.array(hist_cases_gdf[self.CASES_COLUMN])
            past_vars= np.array(hist_variables_df[self.IP_COLUMNS+self.MOBILITY_COLUMNS])
            future_vars = np.array(vars_gdf[self.IP_COLUMNS+self.MOBILITY_COLUMNS])

            # Make prediction for each day
            geo_preds = []
            # Start predicting from start_date, unless there's a gap since last known date
            current_date = min(last_known_date + np.timedelta64(1, 'D'), start_date)
            days_ahead = 0
            while current_date <= end_date:
                # Prepare data
                x_cases = past_cases[-NB_LOOKBACK_DAYS:]
                x_vars = past_vars[-NB_LOOKBACK_DAYS:]
                X = np.concatenate([x_cases.flatten(),
                                    x_vars.flatten()])

                # Make the prediction
                pred = self.model.predict(X.reshape(1, -1))[0]
                pred = max(0, pred)  # Do not allow predicting negative cases
                # Add if it's a requested date
                if current_date >= start_date:
                    geo_preds.append(pred)
                    if verbose:
                        print(f"{current_date.strftime('%Y-%m-%d')}: {pred}")
                else:
                    if verbose:
                        print(
                            f"{current_date.strftime('%Y-%m-%d')}: {pred} - Skipped (intermediate missing daily cases)")

                # Append the prediction and npi's for next day
                # in order to rollout predictions for further days.
                past_cases = np.append(past_cases, pred)
                past_vars = np.append(past_vars, future_vars[days_ahead:days_ahead + 1], axis=0)

                # Move to next day
                current_date = current_date + np.timedelta64(1, 'D')
                days_ahead += 1

            # Create geo_pred_df with pred column
            geo_pred_df = vars_gdf[self.ID_COLUMNS].copy()
            geo_pred_df['PredictedDailyNewCases'] = geo_preds
            geo_pred_dfs.append(geo_pred_df)

        # Combine all predictions into a single dataframe
        pred_df = pd.concat(geo_pred_dfs)

        return pred_df


    def GetCovidPredictions(self, period='30d', nb_lookback_days=20, reduced_train=False, verbose=False, test_several_lookbacks=False):
        """Final function that returns predictions and MAE for each country

            period (str): time period to test the model (either 4 days, 7 days or 30 days)
            nb_lookback_days (int): number of days to consider when looking back at lagged variables
            verbose (bool): whether to run in quiet mode or not """

        #Get our preprocessed data
        print("Starting Preprocessing...")
        data = self.PreProcessing()

        #Split our train and final test set.
        #Train only from August to November if reduced_train.
        if reduced_train:
            training_set = data[(data.Date >= self.train_start_date) & ( data.Date <= self.train_end_date)]

        #Train on the whole data up to January otherwise
        else:
            training_set = data[data.Date <= self.train_end_date]
        test_set = data[(data.Date > self.train_end_date) & ( data.Date <= self.test_end_date)]

        #Predict on test set according to prediction period (either 4 days, 7 days or 30 days)
        if period=='4d':
            start_date='2020-11-30'
            end_date='2020-12-03'
        elif period =='7d':
            start_date='2020-11-30'
            end_date='2020-12-06'
        elif period=='30d':
            start_date='2020-11-30'
            end_date='2020-12-29'


        if test_several_lookbacks:
            global_MAEs=[]
            for lookback_days in [3,5,8,10,15,20,30]:
                #Train the model with/without mobility data and according to the number of lookback days
                print("Starting Training...")
                self.Training(training_set, verbose, lookback_days)

                print("Starting Predicting...")
                predictions = self.Predict(start_date, end_date, training_set, test_set,  lookback_days, verbose)

                #Get our ground_truth
                y_test = test_set[self.ID_COLUMNS+self.CASES_COLUMN]
                y_test = y_test[(y_test.Date >= start_date) & (y_test.Date <= end_date)]

                #Compute the MAE for each country
                MAEs = []
                for g in y_test.GeoID.unique():
                    predicted = np.array(predictions[predictions.GeoID==g]['PredictedDailyNewCases'])
                    ground_truth = np.array(y_test[y_test.GeoID==g][self.CASES_COLUMN])
                    MAEs.append(self.compute_mae(predicted, ground_truth))

                global_MAEs.append(sum(MAEs))

            #Return only global MAEs
            return global_MAEs

        else:
            #Train the model with/without mobility data and according to the number of lookback days
            print("Starting Training...")
            self.Training(training_set, verbose, nb_lookback_days)

            print("Starting Predicting...")
            predictions = self.Predict(start_date, end_date, training_set, test_set,  nb_lookback_days, verbose)

            #Get our ground_truth
            y_test = test_set[self.ID_COLUMNS+self.CASES_COLUMN]
            y_test = y_test[(y_test.Date >= start_date) & (y_test.Date <= end_date)]

            #Compute the MAE for each country
            MAEs = []
            for g in y_test.GeoID.unique():
                predicted = np.array(predictions[predictions.GeoID==g]['PredictedDailyNewCases'])
                ground_truth = np.array(y_test[y_test.GeoID==g][self.CASES_COLUMN])
                MAEs.append(self.compute_mae(predicted, ground_truth))

            #Print global MAE
            print('Global MAE :', sum(MAEs))

            #Get merged dataset with NPIs and Mobility Data
            final_predictions = y_test.merge(predictions[['GeoID','Date','PredictedDailyNewCases']], left_on=['GeoID','Date'], right_on=['GeoID','Date'], how='left')

            return final_predictions, MAEs


    @staticmethod
    def R2_score(true, pred):
        """Computes R2 score (determination coefficient of the prediction)
        true: ground truth
        pred: predicted cases"""
        u=((true - pred) ** 2).sum()
        v=((true - true.mean()) ** 2).sum()
        return 1 - (u/v)
