import pandas as pd
from pandas_profiling import ProfileReport

profile = ProfileReport(pd.read_csv('../data/raw/heart_cleveland_upload.csv'), explorative=True)

profile.to_file("../reports/eda.html")
