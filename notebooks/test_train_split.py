"""
1. start from frame 0 and save 100 frames at interval of 25 into train folder.
2. save each image as scene_video_videonum_framenum.png
3. Put the corresponding features into train.csv file
4. Now do the same for test set from frame 2501 and take 100 frames at interval of 100.
"""
import sys, os
import pandas as pd
from shutil import copyfile

num_images = 0; i = 1

scene = sys.argv[1]
videoname = sys.argv[2]

ann = pd.read_csv('annotations.csv', names=['track_id','xmin','ymin','xmax','ymax','frame','lost','occluded','generated','label'])
ann = ann.drop('track_id', 1); ann = ann.drop('lost', 1); ann = ann.drop('occluded', 1); ann = ann.drop('generated', 1)

train_csv = pd.DataFrame(); test_csv = pd.DataFrame()

while num_images <= 100:
    # save the i th image in train folder by renaming it
    src = os.getcwd()+'/imgs/'+str(i)+'.png'
    dest = os.getcwd()+'/train/'+scene+'_'+videoname+'_'+str(i)+'.png'
    copyfile(src,dest)

    # get all the feature entries corresponding to i th image and append it to train_csv.csv
    df = ann.loc[ann['frame']==i]
    df.reset_index(drop=True, inplace=True)
    length = len(df['frame'])
    for j in range(length):
        df['frame'][j] = 'train/'+scene+'_'+videoname+'_'+str(df['frame'][j])+'.png'
        print("Done for ",j)
    train_csv = train_csv.append(df,ignore_index=True)

    i += 25
    num_images += 1
print("-"*50)
print("Done for training set")
print("-"*50)
i += 1
num_images = 0
while num_images <= 100:
    # save the i th image in test folder by renaming it
    src = os.getcwd()+'/imgs/'+str(i)+'.png'
    dest = os.getcwd()+'/test/'+scene+'_'+videoname+'_'+str(i)+'.png'
    copyfile(src,dest)

    # get all the feature entries corresponding to i th image and append it to test_csv.csv
    df = ann.loc[ann['frame']==i]
    df.reset_index(drop=True, inplace=True)
    length = len(df['frame'])
    for j in range(length):
        df['frame'][j] = 'test/'+scene+'_'+videoname+'_'+str(df['frame'][j])+'.png'
        print("Done for ",j)
    test_csv = test_csv.append(df,ignore_index=True)

    i += 25
    num_images += 1
print("-"*50)
train_csv.to_csv('train_csv.csv',index=False)
test_csv.to_csv('test_csv.csv',index=False)
