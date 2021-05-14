#+ echo=False
import logging
import pickle
import json
import os
import pandas as pd
from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

#+ echo=False
def nan_info(data:pd.DataFrame, columns_name:str) -> int:
    nan_count = len(data[data[columns_name].isna()])
    return nan_count

def outlieres_per_category(data:pd.DataFrame, category:str, outlieres:str, quantile_cutoff = 0.99):

    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize=(len(data[category].unique())*1.3, 
                                                           len(data[category].unique())*1.3))
    data = data.sort_values(by = category)
    ax = sns.boxplot(x = data[category], y = data.price, ax = axs[0][0])
    ax.set_xticklabels(labels = data[category].unique(), rotation = 45)
    ax.set_title(f"{outlieres} per {category} outliers")

    average_price = data.groupby(category).mean()

    ax = sns.barplot(x = average_price.index, y = average_price[outlieres], ax = axs[1][0])
    ax.set_xticklabels(labels = data[category].unique(), rotation = 45)
    ax.set_title(f"Average {outlieres} per {category}")

    for cat in data[category].unique():
        quantile = data[data[category] == cat][outlieres].quantile(quantile_cutoff)
        data = data[~((data[category] == cat) & (data[outlieres] > quantile))]

    ax = sns.boxplot(x = data[category], y = data.price, ax = axs[0][1])
    ax.set_xticklabels(labels = data[category].unique(), rotation = 45)
    ax.set_title(f"{outlieres} per {category} removed outliers")

    average_price = data.groupby(category).mean()
    
    ax = sns.barplot(x = average_price.index, y = average_price[outlieres], ax = axs[1][1])
    ax.set_xticklabels(labels = data[category].unique(), rotation = 45)
    ax.set_title(f"Average {outlieres} per {category} remvoed outliers")

    fig.tight_layout()
    
    plt.show()

def correlation_graph(results:list, 
                      labels:list, 
                      title:str,
                      colors = [(0.1, 0.2, 0.5),  (0.5, 0.2, 0.2)],
                      xlim:list = [75, 100],
                      legend = ["Results", "Labels"]):

    #+ echo=False
    fig, ax = plt.subplots()
    values = [results, labels]
    
    for i in range(len(values)):
        sns.histplot(values[i], bins=range(75, 100, 1), ax=ax, color = colors[i], label = legend[i])
        
    ax.set_xlim(xlim)
    ax.set_title(title)
    ax.legend()
    plt.show()

#+ echo=False
file_json_path = os.getenv('JSON_FILE')

#+ echo=False
model_path = ""
test_data_path = ""
raw_data_dir = ""

#+ echo=False
data = json.load(open(os.getenv("SHARED_FILES"), "r"))
model_path = data["model_dir"]
test_data_path = data["test_set_dir"]
raw_data_dir = data["raw_data_dir"] 

#+ echo=False
metrics = ["Model", "MAE", "RMSE", "R2"]
result_metrics = ["SVM"]
test_set = pd.read_parquet(test_data_path)
raw_set = pd.read_csv(raw_data_dir).drop("Unnamed: 0", errors = "ignore")

#' ## Data Analysis
#+ echo=False
histo = raw_set[os.getenv('LABLE_COL')].plot.hist(title = "Lable Distribution")
plt.show()

#' ### Checking for missing values
#+ echo=False
nan_counts = []
for col in raw_set.columns:
    nan_counts.append(nan_info(raw_set, col))
ax = sns.barplot(y = raw_set.columns, x  = nan_counts)
ax.set_title("Missing Values per Feature (log x-scale)")
ax.set_xscale("log")
plt.show()

#' Designation has to many missing values, thus is not suiteable for a prediction
#' Same for region_1 and region_2
#' The title will also be neglected, as it is hard to draw any correlations from it, since it is individual for each wine

#' See if taster name can be filled by using the corresponding twitter handle
#' They can not, sadly
#+ echo=False
print(f"Number of taster_name that is nan but has a twitter handle {len(raw_set[(raw_set['taster_name'].isna()) & (~raw_set['taster_twitter_handle'].isna())])}" )

#' The plan is to fill the prices by the countries price average
#' Investigating the countries of the missing prices
#+ echo=False
price_to_country = raw_set[raw_set["price"].isna()].country.value_counts()
ax = sns.barplot(y = price_to_country.index, x  = price_to_country.values)
ax.set_title("Missing Prices per Country")
plt.show()

#' Seeing the distribution after outliers are elimnated
#+ echo=False
country_slice = raw_set[raw_set["country"].isin(raw_set[raw_set["price"].isna()].country.unique())].dropna(subset = ["country"])
outlieres_per_category(country_slice, "country", "price", quantile_cutoff = 0.99)

#+ echo=False
y_test = test_set[os.getenv('LABLE_COL')]
x_test = test_set.drop(columns = [os.getenv('LABLE_COL')])

#+ echo=False
model = pickle.load(open(model_path, 'rb'))
results = model.predict(x_test)

#' ## Model Evaluation
#+ echo=False
# result_metrics.append(mean_absolute_error(y_test, results))
# result_metrics.append(mean_squared_error(y_test, results, squared = True))
# result_metrics.append(r2_score(y_test, results))

#' ### Resulting Metrics
#+ echo=False
# print(pd.DataFrame(result_metrics, columns = metrics))
print(f'Mean Absolute Error: {mean_absolute_error(y_test, results)}')
print(f'Mean Squared Error: {mean_squared_error(y_test, results, squared = True)}')
print(f'R2: {r2_score(y_test, results)}')

#+ echo=False
correlation_graph(results, y_test, "Truth and Results correlation")


