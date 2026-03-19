import pandas as pd
import numpy as np
import cv2
import os

# load CSV
df = pd.read_csv(r"C:\Users\pooji\Desktop\Final Year Project\ckextended.csv")

# create folders
os.makedirs("ck_images", exist_ok=True)

for i, row in df.iterrows():
    pixels = np.array(row['pixels'].split(), dtype='uint8')
    img = pixels.reshape(48, 48)

    label = row['emotion']

    folder = f"ck_images/{label}"
    os.makedirs(folder, exist_ok=True)

    cv2.imwrite(f"{folder}/{i}.jpg", img)

print("Images created!")