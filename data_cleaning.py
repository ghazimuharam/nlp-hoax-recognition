import pandas as pd

# Configuration
PATH_DATA = 'Data Latih'
DATASETS = PATH_DATA+'/Data Latih BDC.csv'
BALANCER = 500  # Used to add more data to hoax label, change to 0 if you want make the datasets 1:1

# Open Datasets file
file = pd.read_csv(DATASETS)

file = file.sample(frac=1)

# Balancing Datasets
rslt_val = file[file['label'] == 0]
rslt_hoax = file[file['label'] == 1]
rslt_hoax = rslt_hoax[:len(rslt_val)+BALANCER]

# Create a new dataFrame from Balanced Datasets
newDF = rslt_hoax.append(rslt_val)
newDF = newDF.sample(frac=1)

# Export to csv
print(newDF.to_csv(PATH_DATA+'/Data Clean BDC.csv'))
