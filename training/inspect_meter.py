from preprocessing.load_ukdale import read_meter

df = read_meter("data/ukdale.h5", 1, 5)
print(df["power"].min(), df["power"].max())
