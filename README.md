# PhishIntention

- [PhishIntention]
    - This is the repository for phishintention project
    
    
- [Instructions]
    - Please run phishintention_main.py to get prediction
    ```
      python phishintention_main.py --folder [data folder] --results [xxx.txt]
    ```
    

- [Project structure]
    - src
        - credential_classifier: training scrip for CRP classifier
        - layout_matcher: script for layout matcher and layout heuristic
        - phishpedia: training script for siamese
        - element_detector: training script for element detector

        - element_detector.py: main script for element detector
        - credential.py: main script for CRP classifier
        - layout.py: main script for layout 
        - siamese.py: main script for siamese

        