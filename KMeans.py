import pandas as pd

df=pd.read_csv("C:\\Users\\nitin\\Downloads\\knn_sample_dataset.csv")

df.head()

df.drop(['Class'],axis=1)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit_transform(df)

from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=2,random_state=2)
kmeans.fit(df)

df['cluster']=kmeans.labels_

df



