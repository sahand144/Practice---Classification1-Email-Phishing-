path = r"D:\datasets\New To Work on 3\Email Phishing Dataset.zip"
import zipfile as zip
import pandas as pd
import matplotlib.pyplot as plt

#unzip the file
with zip.ZipFile(path , 'r') as zip_ref:
    print(zip_ref.namelist())
    csv_filename = zip_ref.namelist()[0]
    with zip_ref.open(csv_filename) as file:
        df = pd.read_csv(file)

print(df.head())
print(df.columns)

#look at the data types
print(df.dtypes)

#look at the missing values
print(df.isnull().sum())# we have no missing values

#look at the unique values
print(df.nunique())# we have 2 unique values in the label column 

#look at the descriptive statistics
print(df.describe())

#look at the distribution of the data
print(df.hist(figsize=(10,10)))
plt.show()

#look at the correlation matrix with help of seaborn heatmap plot
import seaborn as sns
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

#now we know our data is all numeric and after applying standardization and split them into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.2, random_state=42)
#before applying standardization lets look at the shape of the training and testing set
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#now we need to apply standardization to the training and testing set
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#now we need to apply the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#now we need to predict the testing set
y_pred = model.predict(X_test)  

#now we need to evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#and for the final part lets visual the reuslt with help of confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='coolwarm')
plt.show()
