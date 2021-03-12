import xml.etree.ElementTree as ET
import re
from sklearn.model_selection import train_test_split
import numpy as np
import os
import shutil

def read_xml(xml_file: str):
    '''read xml file'''
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    list_with_all_types = []

    for boxes in root.iter('object'):

        type = boxes.find('name').text

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
        list_with_all_types.append(type)
    assert len(list_with_all_boxes) == len(list_with_all_boxes)
    return list_with_all_types, list_with_all_boxes

def read_txt(txt_file: str):
    '''read coordinate txt file'''
    contents = [x.strip() for x in open(txt_file).readlines()]
    paths = [x.split('\t')[0] for x in contents]
    coordinates = [x.split('\t')[1] for x in contents]
    types = [x.split('\t')[2] for x in contents]
    classes = [x.split('\t')[3] for x in contents]
    num_imgs = len(set(paths))

    preprocess_coordinates = []
    for coord in coordinates:
        x1, y1, x2, y2 = list(map(float, re.search(r'\((.*?)\)', coord).group(1).split(",")))
        preprocess_coordinates.append([x1, y1, x2, y2])

    assert (len(preprocess_coordinates) == len(classes)) & (len(paths) == len(preprocess_coordinates)) & (len(types) == len(classes))
    return num_imgs, classes, paths, preprocess_coordinates, types

def read_txt_screenshot(txt_file: str):
    '''read labelled txt file'''
    contents = [x.strip() for x in open(txt_file).readlines()]
    paths = [x.split('\t')[0] for x in contents]
    classes = [x.split('\t')[-1] for x in contents]
    num_imgs = len(set(paths))

    assert (len(paths) == len(classes))
    return num_imgs, classes, paths

def split_data(txt_file: str, test_ratio: float):
    '''Train test split'''
    classes_dict = {'credential': 0, 'noncredential': 1}
    num_imgs, classes, paths, preprocess_coordinates, types = read_txt(txt_file)
    all_image_file = paths
    all_img_classes = [classes_dict[x] for x in classes]

    # Remove duplicates
    my_list_unique = set(all_image_file)
    indexes = [all_image_file.index(x) for x in my_list_unique]
    all_image_file = np.asarray(all_image_file)[indexes]
    all_img_classes = np.asarray(all_img_classes)[indexes]
    # print(all_img_classes.shape)

    # Train-test split
    indices = np.arange(len(all_image_file))
    # print(indices.shape)

    _, _,  _, _, idx1, idx2 = train_test_split(np.random.rand(len(all_image_file), 3), all_img_classes, indices,
                                              test_size=test_ratio, random_state=1234, stratify=all_img_classes)


    # Select
    train_image_files, val_image_files = all_image_file[idx1], all_image_file[idx2]
    train_image_classes, val_image_classes = all_img_classes[idx1], all_img_classes[idx2]

    return train_image_files, val_image_files, train_image_classes, val_image_classes


if __name__ == '__main__':
    train_image_files, val_image_files, _, _ = split_data('./data/all_coords.txt', test_ratio=0.1)

    train_img_folder = './data/train_imgs'
    # shutil.rmtree(train_img_folder)
    os.makedirs(train_img_folder, exist_ok=True)
    val_img_folder = './data/val_imgs'
    # shutil.rmtree(val_img_folder)
    os.makedirs(val_img_folder, exist_ok=True)

    train_annot_file = './data/train_coords.txt'
    val_annot_file = './data/val_coords.txt'

    # Copy images over
    # for file in train_image_files:
        # shutil.copyfile('./data/first_round_3k3k/all_imgs/'+file+'.png',
        #                 os.path.join(train_img_folder, file+'.png'))

    # for file in val_image_files:
    #     shutil.copyfile('./data/first_round_3k3k/all_imgs/'+file+'.png',
    #                     os.path.join(val_img_folder, file+'.png'))

    print('Number of training images {}'.format(len(os.listdir(train_img_folder))))
    print('Number of validation images {}'.format(len(os.listdir(val_img_folder))))

    for file in os.listdir(train_img_folder):
        try:
            types, boxes = read_xml(os.path.join('./data/credential_xml', file.replace('.png', '.xml')))
            label = 'credential'
        except:
            types, boxes = read_xml(os.path.join('./data/noncredential_xml/noncredential_xml', file.replace('.png', '.xml')))
            label = 'noncredential'

        for j in range(len(types)):
            with open(train_annot_file, 'a+') as f:
                f.write(file.split('.png')[0] + '\t')
                f.write('(' + ','.join(list(map(str, boxes[j]))) + ')' + '\t')
                f.write(types[j] + '\t')
                f.write(label + '\n')


    for file in os.listdir(val_img_folder):
        try:
            types, boxes = read_xml(os.path.join('./data/credential_xml', file.replace('.png', '.xml')))
            label = 'credential'
        except:
            types, boxes = read_xml(os.path.join('./data/noncredential_xml/noncredential_xml', file.replace('.png', '.xml')))
            label = 'noncredential'

        for j in range(len(types)):
            with open(val_annot_file, 'a+') as f:
                f.write(file.split('.png')[0] + '\t')
                f.write('(' + ','.join(list(map(str, boxes[j]))) + ')' + '\t')
                f.write(types[j] + '\t')
                f.write(label + '\n')

