!sudo add-apt-repository ppa:alessandro-strada/google-drive-ocamlfuse-beta
!sudo apt-get update
!sudo apt-get install google-drive-ocamlfuse

from oauth2client.client import GoogleCredentials
from google.colab import auth, files
import getpass
auth.authenticate_user()
creds = GoogleCredentials.get_application_default()
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
!echo {getpass.getpass()} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
!mkdir -p drive && google-drive-ocamlfuse drive

import sys
sys.path.insert(0, 'drive/ColabNotebooks')

import os
import cv2
import sys
import shutil
import numpy as np

myPath = "drive/ColabNotebooks/images/"

fileNames = [f for f in os.listdir(myPath) if os.path.isfile(os.path.join(myPath, f))]

print(str(len(fileNames)) + ' images loaded...')

dogCount = 0
catCount = 0
trainingSize = 1000
testSize = 500
trainingImgs = []
trainingLabels = []
testImgs = []
testLabels = []
size = 150
dogTrainDir = "drive/ColabNotebooks/datasets3/catvsdog/train/dogs/"
catTrainDir = "drive/ColabNotebooks/datasets3/catvsdog/train/cats/"
dogTestDir = "drive/ColabNotebooks/datasets3/catvsdog/test/dogs/"
catTestDir = "drive/ColabNotebooks/datasets3/catvsdog/test/cats/"

def make_dir(dir):
  if os.path.exists(dir):
    shutil.rmtree(dir)
  os.makedirs(dir)
  
make_dir(dogTrainDir)
make_dir(catTrainDir)
make_dir(dogTestDir)
make_dir(catTestDir)

def get_zeros(number):
  if(number > 10 and number < 100):
    return "0"
  if(number < 10):
    return "00"
  else:
    return ""
  
for i, file in enumerate(fileNames):
  
  if(fileNames[i][0] == "d"):
    dogCount += 1
    img = cv2.imread(myPath+file)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    
    if(dogCount <= trainingSize):
        trainingImgs.append(img)
        trainingLabels.append(1)
        zeros = get_zeros(dogCount)
        cv2.imwrite(dogTrainDir + "dog" + str(zeros) + str(dogCount) + ".jpg", img)
        
    if(dogCount > trainingSize and dogCount <= trainingSize + testSize):
        testImgs.append(img)
        testLabels.append(1)
        zeros = get_zeros(dogCount - 1000)
        cv2.imwrite(dogTestDir + "dog" + str(zeros) + str(dogCount - 1000) + ".jpg", img)
        
  if(fileNames[i][0] == "c"):
    catCount += 1
    img = cv2.imread(myPath+file)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    
    if(catCount <= trainingSize):
        trainingImgs.append(img)
        trainingLabels.append(1)
        zeros = get_zeros(catCount)
        cv2.imwrite(catTrainDir + "cat" + str(zeros) + str(catCount) + ".jpg", img)
        
    if(catCount > trainingSize and catCount <= trainingSize + testSize):
        testImgs.append(img)
        testLabels.append(1)
        zeros = get_zeros(catCount - 1000)
        cv2.imwrite(catTestDir + "cat" + str(zeros) + str(catCount - 1000) + ".jpg", img)
        
  if(dogCount == trainingSize + testSize and catCount == trainingSize + testSize):
    break
        
print("Completed!")

np.savez("drive/training_data.npz", np.array(trainingImgs))
np.savez("drive/training_labels.npz", np.array(trainingLabels))
np.savez("drive/test_data.npz", np.array(testImgs))
np.savez("drive/test_labels.npz", np.array(testLabels))
