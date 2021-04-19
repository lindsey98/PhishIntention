import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import os
import shutil

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('./datasets/gsheetdownload.json', scope)
client = gspread.authorize(creds)

# Find a workbook by name and open the first sheet
# Make sure you use the right name here.
sheet = client.open("test").sheet1

# Extract and print all of the values
list_of_hashes = sheet.get_all_records()
df = pd.DataFrame(list_of_hashes)

# Get breakdown statistics
pedia_tp = []
pedia_fp = []
pedia_unsure = []

intention_tp = []
intention_miss = []
intention_fp = []
intention_unsure = []

for index, row in df.iterrows():
    path = row['date'] + '/' + row['foldername']
    if row['yes'] > 0:
        # phishpedia get it correct
        pedia_tp.append(path)
        if os.path.exists(os.path.join('./datasets/PhishDiscovery/PhishIntention', path)): # intention also get it correct
            intention_tp.append(path)
        else: # intention miss it
            intention_miss.append(path)
    elif row['no'] > 0:
        # mistaken
        pedia_fp.append(path)
        if os.path.exists(os.path.join('./datasets/PhishDiscovery/PhishIntention', path)): # intention also has this FP
            intention_fp.append(path)
        else:
            pass # intention suppress this fp
    elif row['unsure'] > 0:
        # unsure, ignore
        print('ignore')
        pedia_unsure.append(path)
        if os.path.exists(os.path.join('./datasets/PhishDiscovery/PhishIntention', path)): # intention also has this FP
            intention_unsure.append(path)
    else:
        print('Not labelled yet {}'.format(path))

# print out
print('Pedia TP = {}, FP = {}, Unsure = {}'.format(str(len(pedia_tp)), str(len(pedia_fp)), str(len(pedia_unsure))))
print('Intention TP = {}, FP = {}, Miss(FN) = {}, Unsure = {}'.format(str(len(intention_tp)), str(len(intention_fp)), str(len(intention_miss)), str(len(intention_unsure))))

try:
    shutil.rmtree('./datasets/PhishDiscovery/phishintention_miss')
except:
    pass
os.makedirs('./datasets/PhishDiscovery/phishintention_miss', exist_ok=True)
for folder in intention_miss:
    shutil.copytree(os.path.join('./datasets/PhishDiscovery/Phishpedia/', folder),
                    os.path.join('./datasets/PhishDiscovery/phishintention_miss/', folder.split('/')[1]))