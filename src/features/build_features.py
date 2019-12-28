
from pathlib import Path
import pandas as pd




def main():
    directory_in_str = "..\\..\\data\\raw\\Dev\\Audio"
    audioData = pd.DataFrame(columns=['name', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'])
    pathlist = Path(directory_in_str).glob('**/*.csv')
    for path in pathlist:
        # because path is object not string
        audioDict = {}
        path_in_str = str(path)
        singleName = str(path.name.replace('.csv',''))
        df = pd.read_csv(path_in_str, header=None, na_values=['NaN'])
        df = df.fillna(0)
        df = df.mean(axis=1).tolist()
        audioDict = {'name': singleName, '1': df[0], '2': df[1], '3': df[2], '4': df[3], '5': df[4],
                     '6': df[5],
                     '7': df[6], '8': df[7], '9': df[8], '10': df[9], '11': df[10], '12': df[11], '13': df[12]}
        audioData = audioData.append(audioDict, ignore_index=True)


    audioData.to_csv("audio_descriptor_dev.csv", encoding='utf-8', index=False)


    #print(audioData)


if __name__ == "__main__":
    main()