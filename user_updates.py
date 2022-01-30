import pandas as pd
import os
df = pd.DataFrame(columns=["Duration in month","Credit amount"], data=[[1,1]])
new_df = pd.DataFrame(columns=["Duration in month","Credit amount", "abc"], data=[[0, 0, 0]])
#new_df.update(df)
print("new df before" , new_df)
for i in range(0,len(df.columns)):
    new_df[df.columns[i]] = df[df.columns[i]]

writepath = 'update.csv'
mode = 'a' if os.path.exists(writepath) else 'w'
new_df.to_csv(writepath, mode=mode, index=False, header=True)