import pandas as pd

# df = pd.read_csv("./model_history_log.csv")
# df = pd.read_csv("./ddrri_rand_median.csv")
df = pd.read_csv("C:/study/_data/record/santanber_model.csv")

# min_rows = df.loc[df["time"] > "2026-09-04 12:30:18"] #df["test_loss"]+ df["last loss"] <5500
# min_rows = df.loc[ (df["test_loss"] <3300) & (df["last_loss"] <3300)] #df["test_loss"]+ df["last loss"] <5500

# print(min_rows)

# print(min_rows["random_num"])
top = df.nlargest(20, "r2_score", keep='all')

print(top)
print(top["model_structure"])