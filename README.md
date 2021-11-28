# Adult Dataset Data Analysis
Data Analysis of Adult dataset.

# Table of contents
- [

## EDA
### Univariate Analysis
**Histogram**:


![histogram](https://github.com/Dipankar-Medhi/adult_dataset_analysis/blob/master/histogram.jpg)

---
**Box plots**:

![boxplot](https://github.com/Dipankar-Medhi/adult_dataset_analysis/blob/master/boxplot.jpg)

---
**Barplot of categorical features**:

![barplot](https://github.com/Dipankar-Medhi/adult_dataset_analysis/blob/master/catplots.jpg)

### Bivariate Analysis
**Pairplot**:

![pairplot](https://github.com/Dipankar-Medhi/adult_dataset_analysis/blob/master/pairplot.jpg)
---
**Barplot for numerical vs categorical features**:

![barplot](https://github.com/Dipankar-Medhi/adult_dataset_analysis/blob/master/barplot.jpg)

## Data Preprocessing
### Removing outliers and missing values
**IQR**:
```def remove_outlier_IQR(df, field_name):
    iqr = 1.5 * (np.percentile(df[field_name], 75) -
                 np.percentile(df[field_name], 25))
    df.drop(df[df[field_name] > (
        iqr + np.percentile(df[field_name], 75))].index, inplace=True)
    df.drop(df[df[field_name] < (np.percentile(
        df[field_name], 25) - iqr)].index, inplace=True)
    return df

df2 = remove_outlier_IQR(df,'final-wt')
df_final = remove_outlier_IQR(df2, 'hours-per-week')
df_final.shape
```
> (36312, 15)

Boxplot after outliers removal

![outliers_boxplot](https://github.com/Dipankar-Medhi/adult_dataset_analysis/blob/master/outliers_boxplot.jpg)

### Encoding categorical features
- using dummy variables.

### Data preparation for training and testing

```data = pd.concat([cat_df,num_df],axis=1)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = data.drop(columns=['income_<=50K', 'income_>50K'])
y = data['income_<=50K']

scaler = StandardScaler()
scaled_df = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    scaled_df, y, test_size=0.3)
print("X train shape: {} and y train shape: {}".format(
    X_train.shape, y_train.shape))
print("X test shape: {} and y test shape: {}".format(X_test.shape, y_test.shape)
```
> X train shape: (25418, 108) and y train shape: (25418,)
X test shape: (10894, 108) and y test shape: (10894,)

## Model Training
### Random Forest Classifier
![rfc](https://github.com/Dipankar-Medhi/adult_dataset_analysis/blob/master/rfc.jpg)

### Logistic Regression
![lgr](https://github.com/Dipankar-Medhi/adult_dataset_analysis/blob/master/lgr.jpg)

### K Nearest Neighbors
![knn](https://github.com/Dipankar-Medhi/adult_dataset_analysis/blob/master/knn.jpg)

### Naive Bayes
![naiv](https://github.com/Dipankar-Medhi/adult_dataset_analysis/blob/master/naiv.jpg)





