import os

files = os.listdir("dataset/train")

cats = 0
dogs = 0

for file in files:
    if file.startswith("cat"):
        cats += 1
    elif file.startswith("dog"):
        dogs += 1

print("Cats:", cats)
print("Dogs:", dogs)