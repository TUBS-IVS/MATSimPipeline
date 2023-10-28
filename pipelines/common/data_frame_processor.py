class DataFrameProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def process(self):
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates()
        self.df = self.df.reset_index(drop=True)
        return self.df