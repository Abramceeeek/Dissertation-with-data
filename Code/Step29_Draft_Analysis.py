import pandas as pd

df = pd.read_csv('Data/SPX Option Chain/SPX_Options_CLEANED.csv')

snapshot_date = '2018-06-01' 

df_snapshot = df[
    (df['date'] == snapshot_date) &
    (df['cp_flag'].str.lower() == 'c') &
    (df['impl_volatility'] > 0.01) &
    (df['impl_volatility'] < 2.0) &
    (df['strike'] > 100) &
    (df['maturity_days'] <= 365)
].copy()

print(f"Snapshot rows: {len(df_snapshot)}")
print(df_snapshot[['strike', 'maturity_days', 'impl_volatility']].describe())

df_snapshot.to_csv(f'Data/SPX_Snapshot_{snapshot_date}.csv', index=False)
print(f"Saved filtered snapshot to Data/SPX_Snapshot_{snapshot_date}.csv")
