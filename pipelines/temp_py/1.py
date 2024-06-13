from pipelines.common import helpers as h
from utils import settings_values as s

df = h.read_csv(s.ENHANCED_MID_FILE)
# keep only the first 1000 rows
df = df.head(1000)
print(df.head())
df.to_csv(s.ENHANCED_MID_FILE, index=False)
print('done')