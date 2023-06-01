from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np 
from collections import Counter
import seaborn as sns
df = pd.read_csv('fars.csv')  


total = len(df)
pyplot.figure(figsize=(7,5))
g = sns.countplot(x='INJURY_SEVERITY', data=df)
g.set_ylabel('Count', fontsize=14)
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,
            height + 1.5,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14, fontweight='bold')
pyplot.margins(y=0.1)
pyplot.show()

y =np.array(df['INJURY_SEVERITY'].tolist())
