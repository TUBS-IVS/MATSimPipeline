

def distribute_by_weights(self, weights_df: pd.DataFrame, cell_id_col: str, cut_missing_ids: bool = False):
    result = h.distribute_by_weights(self.df, weights_df, cell_id_col, cut_missing_ids)
    self.df = self.df.merge(result[[s.UNIQUE_HH_ID_COL, 'home_loc']], on=s.UNIQUE_HH_ID_COL, how='left')