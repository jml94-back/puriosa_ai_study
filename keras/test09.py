import pandas as pd

# df = pd.read_csv("./model_history_log.csv")
# df = pd.read_csv("./ddrri_rand_median.csv")
df = pd.read_csv("./ddrri_random_num_mean.csv")

# min_rows = df.loc[df["time"] > "2026-09-04 12:30:18"] #df["test_loss"]+ df["last loss"] <5500
min_rows = df.loc[ (df["test_loss"] <2850) & (df["last_loss"] <2850)] #df["test_loss"]+ df["last loss"] <5500

print(min_rows)

print(min_rows["random_num"])
print(min_rows["model_structure"])
print(min_rows["last_loss"])
print(min_rows["test_loss"])