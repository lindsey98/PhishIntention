# PHIND

## PHIND
- This is the official implementation of "Intention Matters: Extracting Static and Dynamic WebpageIntention for Identifying and Explaining Phishing Attacks"
    
## Framework
    
<img src="big_pic/pic.jpg" style="width:2000px;height:350px"/>

```Input```: a screenshot, ```Output```: Phish/Benign, Phishing target
- Step 1: Enter <b>Layout detector</b>, get predicted elements

- Step 2: Enter <b>Siamese</b>
    - If Siamese report no target, ```Return  Benign, None```
    - Else Siamese report a target, Enter step 3 <b>CRP classifier(and HTML heuristic)</b>
       
- Step 3: <b>CRP classifier(and HTML heuristic)</b>
   - If <b>CRP classifier(and heuristic)</b> reports its a CRP page, go to step 5 <b>Return</b>
   - ElIf not a CRP page and havent execute <b>Dynamic analysis</b> before, go to step 4: <b>Dynamic analysis</b>
   - Else not a CRP page but have done <b>Dynamic analysis</b> before, ```Return Benign, None``` 

- Step 4: <b>Dynamic analysis</b>
   - Find login/signup links and click, if reach a CRP page at the end, go back to step 1 <b>Layout detector</b> with updated URL and screenshot
   - Else cannot reach a CRP page, ```Return Benign, None``` 
   
- Step 5: 
    - If reach a CRP + Siamese report target: ```Return Phish, Phishing target``` 
    - Else ```Return Benign, None``` 
    
## Project structure
```
src
    |___ credential_classifier: training script for CRP classifier
    |___ phishpedia: inference script for siamese
    |___ siamese_retrain: training script for siamese 
    |___ element_detector: training script for UI layout detector 
    |___ util: other scripts (chromedriver utilities)
    |___ dynamic: training script for CRP locator 
    |___ detectron2_pedia: training script for logo detector (for Phishpedia not PHIND)
    |___ adv_attack: adversarial attacking script for classification model 
    
    |___ element_detector.py: main script for element detector
    |___ credential.py: main script for CRP classifier
    |___ siamese.py: main script for siamese
    |___ login_finder.py: main script for dynamic login finder
    |___ pipeline_eval.py: evaluation script 

tele: telegram scripts to vote for phishing 

phishintention_config.py: phish-discovery experiment config file for PHIND
phishintention_main.py: phish-discovery experiment evaluation script for PHIND
phishpedia_config.py: phish-discovery experiment config file for Phishpedia
phishpedia_main.py: phish-discovery experiment evaluation script for Phishpedia

```

## Requirements
Linux equipped with GPU

python=3.7 

torch=1.5.1 

torchvision=0.6.0

- Run
```
pip install -r requirements.txt
```
- Install Detectron2 manually, see the [official installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). 

## Instructions
### 1. Download all the model files:
- First download [Siamese model weights](https://drive.google.com/file/d/1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS/view?usp=sharing),
[Logo targetlist](https://drive.google.com/file/d/1_C8NSQYWkpW_-tW8WzFaBr8vDeBAWQ87/view?usp=sharing),
[Brand domain dictionary](https://drive.google.com/file/d/1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1/view?usp=sharing), put them under **src/phishpedia**

- Download [Object detector weights for PHIND](https://drive.google.com/file/d/1HWjE5Fv-c3nCDzLCBc7I3vClP1IeuP_I/view?usp=sharing),
put it under **src/element_detector/output/website_lr0.001/**

- Download [Credential classifier weights](https://drive.google.com/file/d/1igEMRz0vFBonxAILeYMRWTyd7A9sRirO/view?usp=sharing), put it under **src/credential_classifier/output/hybrid/hybrid_lr0.005/**

- Download [Credential locator weights](https://drive.google.com/file/d/1_O5SALqaJqvWoZDrdIVpsZyCnmSkzQcm/view?usp=sharing), put it under **src/dynamic/login_finder/output/lr0.001_finetune**

- (Optional, if you want to run Phishpedia) Download [Object detector weights for Phishpedia](https://drive.google.com/file/d/1tE2Mu5WC8uqCxei3XqAd7AWaP5JTmVWH/view?usp=sharing),
put it under **src/detectron2_pedia/output/rcnn_2/**

### 2. Download all data files
- Download [Phish 30k](https://drive.google.com/file/d/12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g/view?usp=sharing), out of which 4093 are non-credential-requiring phishing, see this [list](https://drive.google.com/file/d/1UVoK-Af3j4ixYy2_jEzG9ZBbYpRkuKFK/view?usp=sharing), shall filter them out when running experiment
- Download [Benign 25k](https://drive.google.com/file/d/1ymkGrDT8LpTmohOOOnA2yjhEny1XYenj/view?usp=sharing) dataset,
unzip and move them to **datasets/**

### 3. Run experiment 
- For general experiment on Phish 25K nonCRP and Benign 25K:
please run evaluation scripts
```
python -m src.pipeline_eval --data-dir [data folder] \
                            --mode [phish|benign] \
                            --write-txt [output_file.txt] \
                            --exp [pedia|intention] \ # evaluate Phishpedia or PHIND
                            --ts [threshold for siamese, suggested to be 0.83]
```

- For phish discovery experiment, the data folder should be organized in [this format](https://github.com/lindsey98/Phishpedia/tree/main/datasets/test_sites):

If you want to run PHIND
```
python phishintention_main.py --folder [data folder] \
                              --results [output_file.txt]
```

If you want to run Phishpedia instead
```
python phishpedia_main.py --folder [data folder] \
                          --results [output_file.txt]
```

## Telegram service to label found phishing (Optional)
### Introduction
- When phishing are reported by the model, users may also want to manually verify the intention of the websites, thus we also developed a telegram-bot to help labeling the screenshot. An example is like this <img src="big_pic/tele.png"/>
- In this application, we support the following command:
```
/start # this will return all the unlabelled data
/get all/date # this will return the statistics for all the data namely how many positive and negatives there are
/classify disagree # this will bring up phishing pages with any disagreement, ie one voted not phishing and one voted phishing for a revote
```
### Setup tele-bot
- 1. Create an empty google sheet for saving the results (foldername, voting results etc.)
- 2. Follow the [guide](https://www.analyticsvidhya.com/blog/2020/07/read-and-update-google-spreadsheets-with-python/) to download JSON file which stores the credential for that particular google sheet, save as **tele/cred.json**
- 3. Go to **tele/tele.py**, Change 
```
token = '[token for telebot]' 
folder = "[the folder you want to label]"
```
[How do I find token for telebot?](https://core.telegram.org/bots#botfather)
- 4. Go to **tele/**, Run **tele.py**

## Miscellaneous
- In our paper, we also implement several phishing detection and identification baselines, see [here](https://github.com/lindsey98/PhishingBaseline)
- We did not include the certstream code in this repo, our code is basically the same as [phish_catcher](https://github.com/x0rz/phishing_catcher), we lower the score threshold to be 40 to process more suspicious websites, readers can refer to their repo for details
- We also did not include the crawling script in this repo, readers can use [Selenium](https://selenium-python.readthedocs.io/), [Scrapy](https://github.com/scrapy/scrapy) or any web-crawling API to crawl the domains obtained from Cerstream, just make sure that the crawled websites are stored in [this format](https://github.com/lindsey98/Phishpedia/tree/main/datasets/test_sites)

