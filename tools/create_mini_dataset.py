import os
import glob
import argparse
import shutil

INDEX = {
    "splicing_connectors": [
        0, # broken cable
        7, # broken cable
        10,# broken cable
        14, # unsymmetric connector 
        24, # missing cable 
        53, # missing one connector 
        54, # two cables
        61, # two cables 
        36, # two short 
        90, # asymmetric cable connection
        104, # asymmetric cable connection
    ],
    "screw_bag": [
        0, # screw with equal length (long)
        24, # screw with equal length (short)
        42, # with additional screw
        53, # with additional screw
        72, # missing screw
        86, # missing screw
        95, # one screw is too short compared with gt
        134, # four hexagon screw
        122, # three flat screw
    ],
    "juice_bottle":[
        0, # missing all labels
        1,
        3,
        12, # missing 100% label, with only cherry label
        16,
        17,
        25, # missing juice
        28, 
        37, # label not in the center
        39,
        40, 
        47, # missing banana label
        51, # label not in the center
        55,
        59, # missing 100% label
        65, # switched labels
        72, # missing main label
        84, # not full bottle
        88, 
        90,
        91, 
        92,
        98, # missing cap
        121, # label logic erro
        122,
        123,
        140
    ],
    "pushpins": [
        0, # two pushping in one compartment
        23,
        30, # missing pushpin 
        31, 
        61, # no pushpin
        63, # broken compartment
    ]
}

def main(args):
    global INDEX
    # there are three folder in the dataset, train, test, and ground_truth
    # delete the ground_truth subfolder
    category_path = os.path.join(args.src, args.category)
    if not os.path.exists(args.src):
        raise ValueError("The source dataset path does not exist.")
    
    # get ground truth folder
    ground_truth_path = os.path.join(category_path, "ground_truth", "logical_anomalies")
    test_path = os.path.join(category_path, "test")

    # get all the test images
    test_images = glob.glob(os.path.join(test_path, "logical_anomalies", "*"))
    category_selected_id = INDEX[args.category]
    for path in test_images:
        id = int(str(os.path.basename(path).split(".")[0]))
        # delete the image if it is not in the selected index
        if id not in category_selected_id:
            os.remove(path)
    
    # remove the unselected ground truth dir
    for id in os.listdir(ground_truth_path):
        if int(id) not in category_selected_id:
            shutil.rmtree(os.path.join(ground_truth_path, id))

if __name__ == "__main__":
    parse = argparse.ArgumentParser("Create mini dataset for testing.")
    parse.add_argument("--category", type=str, default="splicing_connectors", help="category name")
    parse.add_argument("--src", type=str, default="./datasets/MVTec_Loco/mini", help="source dataset path")
    main(args = parse.parse_args())