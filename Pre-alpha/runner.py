import subprocess
import cv2
import sys
import os
import json
import numpy as np
from os import path
from glob import glob
from detect_insulation import InsulationDetection
from detect_sloped_lines import SlopedLinesDetection
from detect_crosses import CrossesDetection


def find_ext(dr, ext):
    return glob(path.join(dr, "*.{}".format(ext)))


def remove_file_type(list_of_names, file_type):
    return [file_name[:-len(file_type)-1] for file_name in list_of_names]


def produce_image_from_pdf(pair_name):
    pdf_name = '{}.pdf'.format(pair_name)
    jpeg_name = '{}.jpeg'.format(pair_name)
    args = ['gs', '-sDEVICE=jpeg', '-dDEVICEWIDTHPOINTS=1920', '-dDEVICEHEIGHTPOINTS=1080', '-dJPEGQ=10', '-dUseCropBox', '-sPageList=1', '-o', jpeg_name, pdf_name]
    sp_result = subprocess.run(args)
    return sp_result.returncode


def write_json(plan_name, position_triple, img_shape):
    with open("{}.ifc.json".format(plan_name)) as json_file:
        data = json.load(json_file)
        ratio = data["ratio"]

    width_in_pixels = (position_triple[1] - position_triple[0]) * img_shape[1]
    height_in_pixels = int(width_in_pixels / ratio)
    height = height_in_pixels / img_shape[0]

    if position_triple[2] < 0.5:
        output = {
            "upper": position_triple[2],
            "lower": min(1., position_triple[2] + height),
            "left": position_triple[0],
            "right": position_triple[1]
        }
    else:
        output = {
            "upper": max(0., position_triple[2] - height),
            "lower": position_triple[2],
            "left": position_triple[0],
            "right": position_triple[1]
        }

    with open("{}.ifc.sections.frame.json".format(plan_name), 'w', encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    assert sys.argv[1] == "-d"
    folder_path = sys.argv[2]

    full_pdf_names = find_ext(folder_path, 'pdf')
    full_json_names = find_ext(folder_path, 'json')

    pdf_names = remove_file_type(full_pdf_names, 'pdf')
    json_names = remove_file_type(full_json_names, 'ifc.json')

    names_of_pairs = set(pdf_names).intersection(json_names)

    for pair_name in names_of_pairs:
        if produce_image_from_pdf(pair_name):
            continue
        gray_image = cv2.imread("{}.jpeg".format(pair_name), cv2.IMREAD_GRAYSCALE)
        gray_image = cv2.normalize(gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        _, gray_image = cv2.threshold(gray_image, 0.5, 1., cv2.THRESH_BINARY)

        det_mechanisms = [InsulationDetection(gray_image), SlopedLinesDetection(gray_image), CrossesDetection(gray_image)]
        candidate_triples = []
        for det_mec in det_mechanisms:
            candidate_triples.append(det_mec.get_candidate_triple())

        index = np.argmin([tri[0] for tri in candidate_triples])
        
        write_json(pair_name, candidate_triples[index], gray_image.shape)
        
        os.remove("{}.jpeg".format(pair_name))


