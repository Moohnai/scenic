import pandas as pd

old_df = pd.read_csv('/home/mona/scenic/scenic/projects/vid2seq/data/youcook2/youcookii_1vid_inf_old.csv')
df = pd.read_csv('/home/mona/scenic/scenic/projects/vid2seq/data/youcook2/youcookii_1vid_inf.csv')

# add features column to df
for i in range(0,len(df)):
    df['features'][i] = old_df['features'][0]

# save the csv file
df.to_csv('/home/mona/scenic/scenic/projects/vid2seq/data/youcook2/youcookii_1vid_inf_new.csv', index=False)