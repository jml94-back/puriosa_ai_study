import pandas as pd

# df = pd.read_csv("./model_history_log.csv")
# df = pd.read_csv("./ddrri_rand_median.csv")
df = pd.read_csv("C:/study/_data/record/ddrri_random_mean_shape.csv")

# min_rows = df.loc[df["time"] > "2026-09-04 12:30:18"] #df["test_loss"]+ df["last loss"] <5500
min_rows = df.loc[ (df["test_loss"] <3300) & (df["last_loss"] <3300)] #df["test_loss"]+ df["last loss"] <5500

# print(min_rows)

# print(min_rows["random_num"])
print(min_rows[["model_structure","last_loss","test_loss","r2_score"]])
print(min_rows["last_loss"])
print(min_rows["test_loss"])

# 588    Dense 128 -> 32 -> 64 -> 128 -> 16 -> 128 -> 1  3296.157227  2946.230713  0.574550

# 709     Dense 128 -> 32 -> 16 -> 128 -> 64 -> 64 -> 1  3031.385742  3092.298584  0.553457

# 780   Dense 128 -> 16 -> 128 -> 128 -> 16 -> 128 -> 1  3262.979248  3292.024902  0.524616
# 796    Dense 128 -> 16 -> 128 -> 64 -> 16 -> 128 -> 1  3150.407959  2948.914551  0.574162
# 811     Dense 128 -> 16 -> 128 -> 32 -> 32 -> 16 -> 1  3277.367920  2905.623047  0.580414
# 880    Dense 128 -> 16 -> 64 -> 16 -> 128 -> 128 -> 1  3152.377930  2949.045410  0.574144
# 908    Dense 128 -> 16 -> 32 -> 128 -> 16 -> 128 -> 1  3184.875732  3121.002197  0.549312
# 946     Dense 128 -> 16 -> 32 -> 16 -> 128 -> 32 -> 1  3197.649658  2945.001709  0.574728
# 1644     Dense 64 -> 32 -> 64 -> 32 -> 16 -> 128 -> 1  3016.368896  3210.370361  0.536407
# 2004     Dense 64 -> 16 -> 16 -> 64 -> 64 -> 128 -> 1  3094.614746  2923.115967  0.577888