# PhishIntention
<div align="center">

![Dialogues](https://img.shields.io/badge/Proctected\_Brands\_Size-277-green?style=flat-square)

</div>
<p align="center">
  <a href="https://www.usenix.org/conference/usenixsecurity22/presentation/liu-ruofan">Paper</a> •
  <a href="https://sites.google.com/view/
phishintention">Website</a> •
  <a href="https://www.youtube.com/watch?v=yU7FrlSJ818">Video</a> •
  <a href="#citation">Citation</a>
</p>

## PhishIntention
- This is the official implementation of "Inferring Phishing Intention via Webpage Appearance and Dynamics: A Deep Vision-Based Approach"USENIX'22 [link to paper](http://linyun.info/publications/usenix22.pdf), [link to our website](https://sites.google.com/view/phishintention/home)

- Existing reference-based phishing detectors:
   - :x: Using screenshots as references does not perfectly capture the brand intention
   - :x: Using logos as references fails to capture the credential-taking intention
     
- The contributions of our paper:
   - :white_check_mark: We propose a referenced-based phishing detection system that captures both brand intention and credential-taking intention. To the best of our knowledge, this is the first work that analyzes both brand intention and credential-taking intentions in a systematic way for phishing detection.
   - :white_check_mark: We address various technical challenges in detecting the intentions by orchestrating multiple deep-learning models. By design, our system is robust against misleading legitimacies and HTML obfuscation attacks.
   - :white_check_mark: We conduct extensive experiments to evaluate our system. The experiments evaluate the overall and step-wise effectiveness, robustness against various adversarial attacks, and usefulness in practice.
   - :white_check_mark: We implement our system with a phishing monitoring system. It reports phishing webpages per day with the highest precision in comparison to state-of-the-art phishing detection solutions.

## Framework
<img src="phishintention/big_pic/Screenshot 2021-08-13 at 9.15.56 PM.png" style="width:2000px;height:350px"/>

```Input```: a screenshot, ```Output```: Phish/Benign, Phishing target
- Step 1: Enter <b>Abstract Layout detector</b>, get predicted elements

- Step 2: Enter <b>Siamese Logo Comparison</b>
    - If Siamese report no target, ```Return  Benign, None```
    - Else Siamese report a target, Enter step 3 <b>CRP classifier</b>
       
- Step 3: <b>CRP classifier</b>
   - If <b>CRP classifier</b> reports its a CRP page, go to step 5 <b>Return</b>
   - ElIf not a CRP page and havent execute <b>CRP Locator</b> before, go to step 4: <b>CRP Locator</b>
   - Else not a CRP page but have done <b>CRP Locator</b> before, ```Return Benign, None``` 

- Step 4: <b>CRP Locator</b>
   - Find login/signup links and click, if reach a CRP page at the end, go back to step 1 <b>Abstract Layout detector</b> with an updated URL and screenshot
   - Else cannot reach a CRP page, ```Return Benign, None``` 
   
- Step 5: 
    - If reach a CRP + Siamese report target: ```Return Phish, Phishing target``` 
    - Else ```Return Benign, None``` 
    
## Updates
- [2023-09-23] ❗❗❗ Somehow the model downloading in setup.sh does not always work because of the git lfs limit 😢 , alternatively, you can download all the models from [this link](https://drive.google.com/file/d/1zw2MViLSZRemrEsn2G-UzHRTPTfZpaEd/view?usp=sharing), and put them into the package. 
    
## Project structure
```
src
    |___ AWL_detector_utils/: scripts for abstract layout detector 
        |__ output/
            |__ website_lr0.001/
                |__ model_final.pth
    |___ crp_classifier_utils/: scripts for CRP classifier
            |__ output/
                |__ Increase_resolution_lr0.005/
                    |__ BiT-M-R50x1V2_0.005.pth.tar
    |___ crp_locator_utils/: scripts for CRP locator 
        |__ login_finder/
            |__ output/
                |__ lr0.001_finetune/
                    |__ model_final.pth
    |___ OCR_siamese_utils/: scripts for OCR-aided Siamese
        |__ demo_downgrade.pth.tar
        |__ output/
            |__ targetlist_lr0.01/
                |__ bit.pth.tar
    |___ util/: other scripts (chromedriver utilities)
    
    |___ phishpedia_siamese/: inference script for siamese (for Phishpedia not PhishIntention)
        |__ domain_map.pkl
        |__ expand_targetlist/
        
    |___ layout_matcher/: deprecated scripts
    
    |___ AWL_detector.py: inference script for AWL detector
    |___ crp_classifier.py: inference script for CRP classifier
    |___ OCR_aided_siamese.py: inference script for OCR-aided siamese
    |___ crp_locator.py: inference script for CRP-Transition locator
    |___ pipeline_eval.py: evaluation script 

phishintention_config.py: phish-discovery experiment config file for PhishIntention
phishintention_main.py: phish-discovery experiment evaluation script for PhishIntention
```

## Instructions
Requirements: 
- CUDA 11

1. Create a local clone of PhishIntention
```bash
git clone https://github.com/lindsey98/PhishIntention.git
```

Run setup 
```bash
cd PhishIntention
chmod +x setup.sh
./setup.sh
```
If you encounter any problem in downloading the models, you can download them manually from here https://huggingface.co/Kelsey98/PhishIntention.

2.
To test a single site

```bash
conda activate myenv
```

```python
from phishintention.phishintention_main import test
import matplotlib.pyplot as plt
from phishintention.phishintention_config import load_config
from phishintention.phishintention_main import element_recognition, phishpedia_classifier_OCR, credential_classifier_mixed_al, driver_loader, dynamic_analysis

# use full model
url = open("phishintention/datasets/test_sites/accounts.g.cdcde.com/info.txt").read().strip()
screenshot_path = "phishintention/datasets/test_sites/accounts.g.cdcde.com/shot.png"
device = 'cuda' # or device = 'cpu'
cfg_path = None # None means use default config.yaml
AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH = load_config(cfg_path, device)

phish_category, pred_target, plotvis, siamese_conf, dynamic, _, pred_boxes, pred_classes = test(url, screenshot_path,
                                                                      AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH)

print('Phishing (1) or Benign (0) ?', phish_category)
print('What is its targeted brand if it is a phishing ?', pred_target)
print('What is the siamese matching confidence ?', siamese_conf)
print('Where are the predicted bounding boxes (in [x_min, y_min, x_max, y_max])?', pred_boxes)
plt.imshow(plotvis[:, :, ::-1])
plt.title("Predicted screenshot with annotations")
plt.show()
```

Or run in the terminal to test a list of sites, copy run.py to your local machine, and run
```bash
python run.py --folder <folder you want to test e.g. phishintention/datasets/test_sites> --results <where you want to save the results e.g. test.txt> --no_repeat
```

## Miscellaneous
- In our paper, we also implement several phishing detection and identification baselines, see [here](https://github.com/lindsey98/PhishingBaseline)
- The phishing discovery crawling script is [here](https://github.com/lindsey98/MyScrapy/tree/main).

## Citation
Please consider citing our work :)
```bibtex
@inproceedings{liu2022inferring,
  title={Inferring Phishing Intention via Webpage Appearance and Dynamics: A Deep Vision Based Approach},
  author={Liu, Ruofan and Lin, Yun and Yang, Xianglin and Ng, Siang Hwee and Divakaran, Dinil Mon and Dong, Jin Song},
  booktitle={30th $\{$USENIX$\}$ Security Symposium ($\{$USENIX$\}$ Security 21)},
  year={2022}
}
```
If you have any issues running our code, you can raise an issue or send an email to liu.ruofan16@u.nus.edu, dcsliny@nus.eud.sg, and dcsdjs@nus.edu.sg

