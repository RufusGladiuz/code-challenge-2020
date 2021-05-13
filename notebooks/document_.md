





## Data Analysis

```
<AxesSubplot:title={'center':'Lable Distribution'},
ylabel='Frequency'>
```

![](document_figure7_1.png)


### Checking for missing values

![](document_figuKre8_1.png)


Designation has to many missing values, thus is not suiteable for a prediction
Same for region_1 and region_2
The title will also be neglected, as it is hard to draw any correlations from it, since it is individual for each wine

```
Empty DataFrame
Columns: [Unnamed: 0, country, description, designation, points,
price, province, region_1, region_2, taster_name,
taster_twitter_handle, title, variety, winery]
Index: []
```



See if taster name can be filled by using the corresponding twitter handle
They can not, sadly

```
Empty DataFrame
Columns: [Unnamed: 0, country, description, designation, points,
price, province, region_1, region_2, taster_name,
taster_twitter_handle, title, variety, winery]
Index: []
```



The plan is to fill the prices by the countries price average
Investigating the countries of the missing prices

![](document_figure11_1.png)


Seeing the distribution after outliers are elimnated

![](document_figure12_1.png)




## Model Evaluation



### Resulting Metrics

```
Mean Absolute Error: 1.7769393641284956
Mean Squared Error: 5.1676625419172915
R2: 0.4278557524657155
```


![](document_figure17_1.png)
