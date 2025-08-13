- This dataset captures key indicators of extroversion and introversion, making it a valuable resource for psychologists, data scientists, and researchers studying social behavior, personality prediction, or data preprocessing techniques.

 
> - Time_spent_Alone: Hours spent alone daily (0–11).  
> - Stage_fear: Presence of stage fright (Yes/No).
> - Social_event_attendance: Frequency of social events (0–10).
> - Going_outside: Frequency of going outside (0–7).
> - Drained_after_socializing: Feeling drained after socializing (Yes/No).
> - Friends_circle_size: Number of close friends (0–15).
> - Post_frequency: Social media post frequency (0–10).
> - Personality: Target variable (Extrovert/Introvert).*


```python
#Install the dependencies
!pip install pandas matplotlib numpy seaborn
!pip install scikit-learn
```

    Requirement already satisfied: pandas in c:\users\user\appdata\local\programs\python\python313\lib\site-packages (2.2.3)
    Requirement already satisfied: matplotlib in c:\users\user\appdata\local\programs\python\python313\lib\site-packages (3.10.3)
    Requirement already satisfied: numpy in c:\users\user\appdata\local\programs\python\python313\lib\site-packages (2.2.6)
    Requirement already satisfied: seaborn in c:\users\user\appdata\local\programs\python\python313\lib\site-packages (0.13.2)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\user\appdata\roaming\python\python313\site-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in c:\users\user\appdata\local\programs\python\python313\lib\site-packages (from pandas) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in c:\users\user\appdata\local\programs\python\python313\lib\site-packages (from pandas) (2025.2)
    Requirement already satisfied: contourpy>=1.0.1 in c:\users\user\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (1.3.2)
    Requirement already satisfied: cycler>=0.10 in c:\users\user\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\user\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (4.58.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in c:\users\user\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (1.4.8)
    Requirement already satisfied: packaging>=20.0 in c:\users\user\appdata\roaming\python\python313\site-packages (from matplotlib) (25.0)
    Requirement already satisfied: pillow>=8 in c:\users\user\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (11.2.1)
    Requirement already satisfied: pyparsing>=2.3.1 in c:\users\user\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (3.2.3)
    Requirement already satisfied: six>=1.5 in c:\users\user\appdata\roaming\python\python313\site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)


    
    [notice] A new release of pip is available: 25.0.1 -> 25.1.1
    [notice] To update, run: python.exe -m pip install --upgrade pip


    Requirement already satisfied: scikit-learn in c:\users\user\appdata\local\programs\python\python313\lib\site-packages (1.6.1)
    Requirement already satisfied: numpy>=1.19.5 in c:\users\user\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (2.2.6)
    Requirement already satisfied: scipy>=1.6.0 in c:\users\user\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (1.15.3)
    Requirement already satisfied: joblib>=1.2.0 in c:\users\user\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (1.5.1)
    Requirement already satisfied: threadpoolctl>=3.1.0 in c:\users\user\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (3.6.0)


    
    [notice] A new release of pip is available: 25.0.1 -> 25.1.1
    [notice] To update, run: python.exe -m pip install --upgrade pip



```python
#import the dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier  # for classification
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

```


```python
#load the dataset and display the first rows
data=pd.read_csv(r"C:\Users\user\Downloads\personality_datasert.csv")
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time_spent_Alone</th>
      <th>Stage_fear</th>
      <th>Social_event_attendance</th>
      <th>Going_outside</th>
      <th>Drained_after_socializing</th>
      <th>Friends_circle_size</th>
      <th>Post_frequency</th>
      <th>Personality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.0</td>
      <td>No</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>No</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>Extrovert</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.0</td>
      <td>Yes</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>Introvert</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.0</td>
      <td>Yes</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Yes</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>Introvert</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>No</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>No</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>Extrovert</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>No</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>No</td>
      <td>8.0</td>
      <td>5.0</td>
      <td>Extrovert</td>
    </tr>
  </tbody>
</table>
</div>




```python
#make a copy of the dataset

df = data.copy()
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time_spent_Alone</th>
      <th>Stage_fear</th>
      <th>Social_event_attendance</th>
      <th>Going_outside</th>
      <th>Drained_after_socializing</th>
      <th>Friends_circle_size</th>
      <th>Post_frequency</th>
      <th>Personality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.0</td>
      <td>No</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>No</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>Extrovert</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.0</td>
      <td>Yes</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>Introvert</td>
    </tr>
  </tbody>
</table>
</div>




```python
#change columns to lower case
df.columns = df.columns.str.lower()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time_spent_alone</th>
      <th>stage_fear</th>
      <th>social_event_attendance</th>
      <th>going_outside</th>
      <th>drained_after_socializing</th>
      <th>friends_circle_size</th>
      <th>post_frequency</th>
      <th>personality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.0</td>
      <td>No</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>No</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>Extrovert</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.0</td>
      <td>Yes</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Yes</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>Introvert</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.0</td>
      <td>Yes</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Yes</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>Introvert</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>No</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>No</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>Extrovert</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>No</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>No</td>
      <td>8.0</td>
      <td>5.0</td>
      <td>Extrovert</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check the shape of the dataset. Dataset has 2900 rows and 8 columns

df.shape
```




    (2900, 8)




```python
# Getting the dataset statistics
df.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>time_spent_alone</th>
      <td>2900.0</td>
      <td>4.505816</td>
      <td>3.441180</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.000000</td>
      <td>7.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>social_event_attendance</th>
      <td>2900.0</td>
      <td>3.963354</td>
      <td>2.872608</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.963354</td>
      <td>6.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>going_outside</th>
      <td>2900.0</td>
      <td>3.000000</td>
      <td>2.221597</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.000000</td>
      <td>5.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>friends_circle_size</th>
      <td>2900.0</td>
      <td>6.268863</td>
      <td>4.232340</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>5.000000</td>
      <td>10.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>post_frequency</th>
      <td>2900.0</td>
      <td>3.564727</td>
      <td>2.893587</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.000000</td>
      <td>6.0</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#checking for datatypes

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2900 entries, 0 to 2899
    Data columns (total 8 columns):
     #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
     0   time_spent_alone           2900 non-null   float64
     1   stage_fear                 2900 non-null   object 
     2   social_event_attendance    2900 non-null   float64
     3   going_outside              2900 non-null   float64
     4   drained_after_socializing  2900 non-null   object 
     5   friends_circle_size        2900 non-null   float64
     6   post_frequency             2900 non-null   float64
     7   personality                2900 non-null   object 
    dtypes: float64(5), object(3)
    memory usage: 181.4+ KB



```python
#Checking for nulls

df.isnull().sum()

```




    time_spent_alone             0
    stage_fear                   0
    social_event_attendance      0
    going_outside                0
    drained_after_socializing    0
    friends_circle_size          0
    post_frequency               0
    personality                  0
    dtype: int64




```python
#check for duplicates
int(df.duplicated().sum())
```




    402




```python
# Remove the duplicates from the dataset.
# Dropping the duplicates will help reduce bias or overfitting in the model.

df = df.drop_duplicates()
```


```python
# checking if the duplicates have been dropped
int(df.duplicated().sum())
```




    0




```python
df.head(2)
numerical_columns = ['time_spent_alone', 'social_event_attendance', 'going_outside', 'friends_circle_size',	'post_frequency']
categorical_cols = ['stage_fear','drained_after_socializing' ]
target_column = ['personality']
```


```python
# Checking for outliers in our dataset
df.head(2)

plt.figure(figsize = (15,10))
for i in range(len(numerical_columns)):
    plt.subplot(3,3,i+1)
    sns.boxplot(y=numerical_columns[i], data = df)
    plt.title(numerical_columns[i])
plt.tight_layout
plt.show()

# The dataset has no outliers
```


    
![png](output_14_0.png)
    



```python
# Checking for dataset distribution
# Drop target and label columns
numerical_columns = df.drop(columns=['stage_fear', 'drained_after_socializing', 'personality'], axis=1)

# Determine grid size (e.g., 3 rows × 3 columns for up to 9 features)
n_cols = 3
n_rows = (len(numerical_columns.columns) + n_cols - 1) // n_cols  # Ceiling division

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()  # Flatten in case of multiple rows

# Plot each KDE on a subplot
for idx, column in enumerate(numerical_columns.columns):
    sns.kdeplot(x=column, data=df, ax=axes[idx], fill=True, color='blue')
    axes[idx].set_title(f"{column} distribution")
    axes[idx].set_xlabel(column)
    axes[idx].set_ylabel("Density")

# Hide any unused subplots
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
```


    
![png](output_15_0.png)
    



```python
categorical_cols = ['stage_fear','drained_after_socializing' ]

missing_cols = [col for col in categorical_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"The following columns are missing in 'data': {missing_cols}")

OHE = OneHotEncoder(handle_unknown = 'ignore',sparse_output = False, drop = 'first').set_output(transform = 'pandas')
encoded_o = OHE.fit_transform(df[categorical_cols])

df = df.drop(columns = categorical_cols)
df_encoded = pd.concat([df,encoded_o], axis = 1)

df_encoded.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time_spent_alone</th>
      <th>social_event_attendance</th>
      <th>going_outside</th>
      <th>friends_circle_size</th>
      <th>post_frequency</th>
      <th>personality</th>
      <th>stage_fear_Yes</th>
      <th>drained_after_socializing_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>Extrovert</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>Introvert</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>Introvert</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>Extrovert</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>8.0</td>
      <td>5.0</td>
      <td>Extrovert</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Machine learning using DecisionTrees

x = df_encoded.drop(columns = 'personality')
y =df_encoded['personality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Scalling the dataset
scaler = StandardScaler()
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train),columns = x.columns)
x_test_scaled = pd.DataFrame(scaler.fit_transform(x_test),columns = x.columns)

print(f'x_train_scaled: {x_train_scaled.shape}, x_test_scaled: {x_test_scaled.shape}')
print(f'y_train: {y_train.shape}, y_test: {y_test.shape}')


```

    x_train_scaled: (1998, 7), x_test_scaled: (500, 7)
    y_train: (1998,), y_test: (500,)



```python
d_tree = DecisionTreeClassifier(criterion="entropy", max_depth = 3)

#Training the model
d_model = d_tree.fit(x_train_scaled,y_train)

# Predict
y_pred = d_model.predict(x_test_scaled)
report = classification_report(y_test,y_pred)
print(report)

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f'\n Confusion Matrix\n {cm}')
```

                  precision    recall  f1-score   support
    
       Extrovert       0.93      0.90      0.92       279
       Introvert       0.88      0.92      0.90       221
    
        accuracy                           0.91       500
       macro avg       0.91      0.91      0.91       500
    weighted avg       0.91      0.91      0.91       500
    
    
     Confusion Matrix
     [[251  28]
     [ 18 203]]



```python
# Machine learning using Random Forest
r_forest = RandomForestClassifier(random_state=42)
r_forest.fit(x_train_scaled, y_train)

# Predict
y_pred = r_forest.predict(x_test_scaled)
report = classification_report(y_test,y_pred)
print(report)

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f'\n Confusion Matrix\n {cm}')

#[[TP FN]
#[FP TN]]

# Plot using sklearn built-in ConfusionMatrixDisplay
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay(cm).plot()
plt.show()
```

                  precision    recall  f1-score   support
    
       Extrovert       0.90      0.88      0.89       279
       Introvert       0.85      0.88      0.87       221
    
        accuracy                           0.88       500
       macro avg       0.88      0.88      0.88       500
    weighted avg       0.88      0.88      0.88       500
    
    
     Confusion Matrix
     [[246  33]
     [ 27 194]]



    
![png](output_19_1.png)
    



```python
# Machine learning using GBC
from sklearn.model_selection import cross_val_score
g_boost =GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
g_boost.fit(x_train_scaled, y_train)
y_pred_gbc = g_boost.predict(x_test_scaled)
print("Gradient Boosting Report:\n", classification_report(y_test, y_pred_gbc))

#cross validation 
scores = cross_val_score(g_boost, x_test_scaled, y_test, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", scores)
print("Average Accuracy:", scores.mean())

#Confusion matrix
cm = confusion_matrix(y_test, y_pred_gbc)
print(f'\n Confusion Matrix\n {cm}')

# Plot using sklearn built-in ConfusionMatrixDisplay
labels = ['Extrovert', 'Introvert']
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(6,4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Oranges')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix Heatmap')
plt.show()
```

    Gradient Boosting Report:
                   precision    recall  f1-score   support
    
       Extrovert       0.93      0.90      0.91       279
       Introvert       0.88      0.92      0.90       221
    
        accuracy                           0.91       500
       macro avg       0.90      0.91      0.91       500
    weighted avg       0.91      0.91      0.91       500
    
    Cross-Validation Scores: [0.92 0.86 0.91 0.87 0.92]
    Average Accuracy: 0.8960000000000001
    
     Confusion Matrix
     [[250  29]
     [ 18 203]]



    
![png](output_20_1.png)
    



```python
# Using KNN Classifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn_fit = knn.fit(x_train_scaled, y_train)
print(knn_fit)

y_knn_pred = knn_fit.predict(x_test_scaled)

#Evaluate the model
report = classification_report(y_test, y_knn_pred)
print(f'\n Classification report\n',report)

#A confusion matrix is a summary table to evaluate the performance of a classification model
#It shows how many predictions the model got right and how many it got wrong, by comparing the actual (test) labels with the predicted labels.

cm = confusion_matrix(y_test, y_knn_pred)
print(f'\n Confusion Matrix\n {cm}')

# Plot using sklearn built-in ConfusionMatrixDisplay
labels = ['Extrovert', 'Introvert']
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(6,4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Oranges')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix Heatmap')
plt.show()


```

    KNeighborsClassifier()
    
     Classification report
                   precision    recall  f1-score   support
    
       Extrovert       0.93      0.90      0.91       279
       Introvert       0.87      0.91      0.89       221
    
        accuracy                           0.90       500
       macro avg       0.90      0.91      0.90       500
    weighted avg       0.91      0.90      0.90       500
    
    
     Confusion Matrix
     [[250  29]
     [ 19 202]]



    
![png](output_21_1.png)
    


## Stacking Classifier  

> - Stacking combines predictions of multiple models and feeds them into a meta-model (usually simpler like Logistic Regression) to make the final prediction.


```python
# Base models
estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=10, random_state=42))
]

# Meta-model
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Train
stack_fit = stack.fit(x_train_scaled, y_train)

# Predict
y_pred_stack = stack_fit.predict(x_test_scaled)
print("Stacking Classifier Report:\n", classification_report(y_test, y_pred_stack))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_stack)
print(f'\n Confusion Matrix\n {cm}')

# Plot using sklearn built-in ConfusionMatrixDisplay
labels = ['Extrovert', 'Introvert']
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(6,4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Oranges')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix Heatmap')
plt.show()

```

    Stacking Classifier Report:
                   precision    recall  f1-score   support
    
       Extrovert       0.93      0.90      0.92       279
       Introvert       0.88      0.92      0.90       221
    
        accuracy                           0.91       500
       macro avg       0.91      0.91      0.91       500
    weighted avg       0.91      0.91      0.91       500
    
    
     Confusion Matrix
     [[251  28]
     [ 18 203]]



    
![png](output_23_1.png)
    

