import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from sklearn.model_selection import  train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix

# ---------------------------  DATA VISUALIZE  --------------------------- #

#col1 = ["job","marital","education","default","housing","loan","duration","campaign","pdays","previous","poutcome","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed","y"]
col1 = ["job","marital","education","default","housing","loan", "y"]
col2 = ["job","marital","education","default","housing","loan", "y"]


#datafile1 = pd.read_csv('D:/Prodigy_Projects/Task3/dataset/bank-marketing/bank-additional/bank-additional/bank-additional-full.csv')
datafile1 = pd.read_csv('F:/Prodigy_Projects/Task3/dataset/bank-marketing/bank-additional/bank-additional/bank-additional.csv')    

df1 = pd.DataFrame(datafile1, columns=col1)
df2 = pd.DataFrame(datafile1, columns=col2)

df3 = df1.query('y=="yes"')
df4 = df1.query('y=="no"')


def bar_plot(variable:str):
    """
    input: variable ex: "Age", "job type"
    output: barplot & value count
    """
      
     # Create a bar chart for the user ages
    var_group3 = df3.groupby(variable)
    print("This is Var Grp", var_group3)
    var_count3 = var_group3[variable].count()
    print("This is Var Count", var_count3)
    #var_chart3 = var_count3.plot(kind='bar')
    #plt.show()

    key1 = var_count3.keys()
    print(">>>>>", key1)
    
     # Create a bar chart for the user ages
    var_group4 = df4.groupby(variable)
    print("This is Var Grp", var_group4)
    var_count4 = var_group4[variable].count()
    print("This is Var Count", var_count4)
    #var_chart4 = var_count4.plot(kind='bar')
    #plt.show()
    key2 = var_count4.keys()
    print(">>>>>", key2)
    
    
    df = pd.merge(var_count3, var_count4, on=key1)
    print("This is Merged Plot: ", df)

    
    #df.plot(kind='bar')
    df.set_index('key_0').plot(kind='bar')
    plt.show()


#["age","job","marital","education","default","housing","loan","duration","campaign","pdays","previous","poutcome","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed","y"
 
category1 = ["job", "marital"]# "education", "housing", "loan"] 

for c in category1:
    bar_plot(c)
    
# ----------------------  DECISION TREE CLASSIFIER  ------------------------ #
#preliminary operations
print('Basic Decision Tree Classifier Operations')
#sns.pairplot(datafile1, hue='col')     #Pairplot
print('This is to check info. in df \n:', datafile1.info())                #check info df   
print('This helps us to describe df \n:', datafile1.describe())            #check stats df 
print('This is to print first 5 rows of df \n:', datafile1.head())         #check head df 
print('This is to print last 5 rows of df \n:', datafile1.tail())          #check tail df

#now splitting the dataset
X = datafile1["age"]
X = np.array(X).reshape(-1, 1)
y = datafile1["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3)

#fitting into the model
tree = DecisionTreeClassifier()
print(tree.fit(X_train, y_train))

#now making predictions
pred = tree.predict(X_test)
print('This is the prediction', pred)

#Evaluatatin
print('This is Evaluation of the Decision Tree Model')
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))