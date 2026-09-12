import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')
print(df.head())
print(df.describe())

sns.countplot(x='survived', data=df)
plt.savefig('titanic_survival.png')

sns.pairplot(df.select_dtypes(include=['number']))
plt.savefig('titanic_pairplot.png')
