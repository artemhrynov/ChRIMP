def aggregate_top_k(df, conditions: dict[str, callable] ,k=1, filtering_strategy="keep_below_k", return_df=False):
    strategy_list = ["keep_below_k", "keep_k_first", "keep_only_kth"]

    assert filtering_strategy in strategy_list, f"Filtering strategy can only be one of {strategy_list}"

    if filtering_strategy == "keep_k_first": # Here, we keep the k-smallest predictions
        df_top_k = (
            df.sort_values("beam_index") # ensure it is sorted
            .groupby(["input", "gold_output"], as_index=False, group_keys=False)
            .head(k) # keep the k smallest per group
        )
    
    elif filtering_strategy == "keep_only_kth": # Here, we keep the predictions that have a beam-index of exactly k-1
        df_top_k = df[df["beam_index"] == (k-1)].copy()

    elif filtering_strategy == "keep_below_k": # Here, we keep the predictions that have a beam-index below k
        df_top_k = df[df["beam_index"] < k].copy()

    else:
        raise NotImplementedError(f"Strategy {filtering_strategy} not implemented yet")
        

    agg_dict = {col: func for col, func in conditions.items()}
    checks = df_top_k.groupby(['input', 'gold_output']).agg(agg_dict)
    
    if return_df:
        return df_top_k[checks.all(axis=1)].reset_index(drop=True)
    else:
        return checks.all(axis=1).sum()
