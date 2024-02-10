import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

train = pd.read_csv("01_import_data/output/train.csv")
test =  pd.read_csv("01_import_data/output/test.csv")

# EDA

train.head()
train.shape # 165034 observaciones

test.head()
test.shape # trae 13 pues no inclute la variable de resultado

 # Mis variables son 14 con el id siendo el que necesito
train.info() 
test.info() 

'''
 0   id               165034 non-null  int64
 1   CustomerId       165034 non-null  int64 # quitar
 2   Surname          165034 non-null  object # rm for firs pero intentar saber si es imigrante
 3   CreditScore      165034 non-null  int64 
 4   Geography        165034 non-null  object # One hot encoding
 5   Gender           165034 non-null  object # One hot encoding
 6   Age              165034 non-null  float64
 7   Tenure           165034 non-null  int64
 8   Balance          165034 non-null  float64
 9   NumOfProducts    165034 non-null  int64 
 10  HasCrCard        165034 non-null  float64 # Factor
 11  IsActiveMember   165034 non-null  float64 # Factor
 12  EstimatedSalary  165034 non-null  float64
 13  Exited           165034 non-null  int64 <- var de interes si es 1 se sale 0 de lo contrario

'''

train['Geography'].unique() # Gente de Francia, España y Alemania

train['Geography'].value_counts(normalize = True) # La gran mayorian 57% son Franceses
test['Geography'].value_counts(normalize = True) # Proporcion practicamente igual en el test

train['Exited'].value_counts(normalize=True) # algo de desbalanbce 78% No salieron

train['NumOfProducts'].value_counts(normalize=True) # 51% tienen dos productos, 46% tienen un producto. 
test['NumOfProducts'].value_counts(normalize=True) # 51% tienen dos productos, 46% tienen un producto. 

# El grupo de test y train se parecen un monton 
train['Gender'].unique() # Gente de Francia, España y Alemania


# ***** Descriptivas basicas ***** #
train[['CreditScore',  'Balance', 'Tenure', 'EstimatedSalary']].describe() # Media alta de salsarios 


# ***** Graficos interesantes **** #

sns.set(style="darkgrid")
sns.displot(data = train, x = 'CreditScore') # Parece una dist normal de los ratings crediticios centrado al rededor de 660, valores extremos pues es un upopoer bound. 
#plt.show()

sns.histplot(data = train, x = 'EstimatedSalary') # Tambien parece una distribucion normal aunque alfo sesgada a la derecha
#plt.show()

sns.histplot(data = train, x = 'NumOfProducts') # Gran mayoria 1 o 2 productos, pocos con 3 o 4
#plt.show()

sns.barplot(data = train, x = 'Exited', y = 'Balance', estimator=np.mean) # los que salen tienen mas balance. 
#plt.show()

# Modifico datos

# Cambiar sexo a 1 y 0

train['Gender'] = (train['Gender'] == 'Male').astype(int)
test['Gender'] = (test['Gender'] == 'Male').astype(int)

# Cambiar paises a onehot encoding

train= pd.get_dummies(data = train, columns= ['Geography'], dtype=int)

test= pd.get_dummies(data = test, columns= ['Geography'], dtype=int)

train.info() 
test.info() 

# Quitar variable de nombre por ahora y la de customer ID

train.drop(['CustomerId', 'Surname'], inplace=True, axis=1)
test.drop(['CustomerId', 'Surname'], inplace=True, axis=1)

train.to_csv("02_prepare_data/output/train.csv", index=False)
test.to_csv("02_prepare_data/output/test.csv", index=False)
