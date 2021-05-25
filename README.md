# PHIND

## PHIND
- This is the repository for phishintention project
    
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
    |___ credential_classifier: training scrip for CRP classifier
    |___ layout_matcher: script for layout matcher and layout heuristic
    |___ phishpedia: training script for siamese
    |___ element_detector: training script for element detector
    |___ util: other scripts (chromedriver utilities)
    
    |___ element_detector.py: main script for element detector
    |___ credential.py: main script for CRP classifier
    |___ layout.py: main script for layout 
    |___ siamese.py: main script for siamese
    |___ login_finder.py: main script for dynamic login finder

phishintention_config.py: config file for PHIND
phishintention_main.py: main script for PHIND
phishpedia_config.py: config file for Phishpedia
phishpedia_main.py: main script for Phishpedia

```

        
## Requirements
- Linux machine equipped with GPU 

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
-- If you want to run PHIND
```
python phishintention_main.py --folder [data folder] \
                              --results [output_file.txt]
```
-- If you want to run phishpedia instead
```
python phishpedia_main.py --folder [data folder] \
                          --results [output_file.txt]
```



