import os
import sys
import csv
import joblib 
from tqdm import tqdm 
import cv2

def load_data():
    file = "../input/words.txt"
    gtTexts = []
    filenames = []
    f = open("../input/words.txt")
    for line in f.readlines():
        if not line or line[0] == "#":
            continue
        
        linesplit = line.strip().split(" ")
        assert len(linesplit) >= 9
        gtText = linesplit[8]
        # print(linesplit)
        filename = linesplit[0]
        filenamesplit = filename.split("-")

        # So the final filename now becomes
        # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
        final_filename = f"{filenamesplit[0]}/{filenamesplit[0]}-{filenamesplit[1]}/{filename}.png"
        if not os.path.getsize(f"../input/words/{final_filename}"):
            continue
        gtTexts.append(linesplit[8])
        filenames.append(final_filename)
    return gtTexts, filenames
        # print(final_filename, linesplit[8])
        # print(os.path.exists(f"../input/words/{final_filename}"), f"../input/words/{final_filename}",  gtText)

def write_to_disk():
    filename = "../data/train.csv"
    gtTexts, filenames = load_data()
    print(gtTexts[:5])
    with open(filename, "a") as f:
        for i in tqdm(range(len(gtTexts))):
            gtText = gtTexts[i]
            filename = f"../input/words/{filenames[i]}"
            image = cv2.imread(filename)
            joblib.dump(image, f"../data/{i}.pkl")
            f.write(f"../data/{i}.pkl,{gtText}\n")


# print(line[:5])
if __name__ == "__main__":
    write_to_disk()