import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

adult = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
df = adult.frame

print(df.head())
df['age'].hist(bins=30)
plt.savefig('adult_age.png')
print(df['class'].value_counts())
