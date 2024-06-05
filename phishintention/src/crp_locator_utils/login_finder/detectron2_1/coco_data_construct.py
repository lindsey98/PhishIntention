
import re
import json


class CreateCocoJson():
    
    def __init__(self, imgpath, jsonpath, category_dict, annot=True, annot_path=None):
        
        self.imgpath = imgpath
        self.jsonpath = jsonpath
        self.annot = annot
        self.annot_path = annot_path
        self.category_dict = category_dict
        if annot:
            self.datadict = {'images':[], 'annotations':[], "categories": category_dict}
        else:
            self.datadict = {'images':[], "categories": category_dict}
            
        self.annotations, self.filenames, self.coords, self.cls = self._read_coords()
        
    def _read_coords(self):
        
        annotations = open(self.annot_path).readlines()
        filenames = [x.strip().split('\t')[0] for x in annotations]
        coords = [x.strip().split('\t')[1] for x in annotations]
        cls = [x.strip().split('\t')[2] for x in annotations]
        return annotations, filenames, coords, cls
    
    def _write_images(self):
        
        image_id = 0
        for path in os.listdir(self.imgpath):
            imgpath = os.path.join(self.imgpath, path)
            img = cv2.imread(imgpath)
            img_height, img_width, _ = img.shape

            image = {
                "file_name": path,
                "height": img_height,
                "width": img_width,
                "id": image_id,
            }

            self.datadict["images"].append(image)
            if self.annot:
                self._write_annotations(path, image_id)

            image_id += 1
        
    def _write_annotations(self, path, image_id):
        

        for k, file in enumerate(self.filenames):
            if file == path.split('.png')[0]: # correponding annotations for path
                x1, y1, x2, y2 = list(map(int, re.search(r'\((.*?)\)', self.coords[k]).group(1).split(",")))
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                
                # find corresponding category id
                category_id = next(item['id'] for item in self.category_dict if item["name"] == self.cls[k])
#                 print(category_id)
                id_annot = len(self.datadict["annotations"]) + 1 #id field must start with 1

                ann = {
                    "area": width * height,
                    "image_id": image_id,
                    "bbox": [x1, y1, width, height],
                    "category_id": category_id,
                    "id": id_annot, # id for box, need to be continuous
                    "iscrowd": 0
                    }

                self.datadict["annotations"].append(ann)
                    
    
    def main(self):
        self._write_images()
        with open(self.jsonpath, 'wt', encoding='UTF-8') as f:
            json.dump(self.datadict, f)



