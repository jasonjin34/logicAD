import csv, sys, os
from operator import add
import argparse
import pdb

VISA0 = {
     # iauroc, pauroc, if1, pf1
    "candle":     [95.4, 88.9, 89.4, 22.5],
    "capsules":   [85.0, 81.6, 83.9, 9.2],
    "cashew":     [92.1, 84.7, 88.4, 13.2],
    "chewinggum": [96.5, 93.3, 94.8, 41.1],
    "fryum":      [80.3, 88.5, 82.7, 22.1],
    "macaroni1":  [76.2, 70.9, 74.2, 7.0],
    "macaroni2":  [63.7, 59.3, 69.8, 1.0],
    "pcb1":       [73.6, 61.2, 71.0, 2.4],
    "pcb2":       [51.2, 71.6, 67.1, 4.7],
    "pcb3":       [73.4, 85.3, 71.0, 10.3],
    "pcb4":       [79.6, 94.4, 74.9, 32.0],
    "pipe_fryum": [69.7, 75.4, 80.7, 12.3],
    "average":    [78.1, 79.6, 79.0, 14.8]
}

VISA4 = {
    # iauroc, pauroc, if1, pf1
    "candle":     [95.1, 97.8, 88.9, 43.0],
    "capsules":   [86.8, 97.1, 86.0, 59.8],
    "cashew":     [95.2, 98.7, 91.6, 62.3],
    "chewinggum": [97.7, 98.5, 95.7, 65.2],
    "fryum":      [90.8, 97.1, 88.9, 56.5],
    "macaroni1":  [85.2, 97.0, 78.2, 33.8],
    "macaroni2":  [70.9, 97.3, 73.1, 35.1],
    "pcb1":       [88.3, 98.1, 83.1, 50.9],
    "pcb2":       [67.5, 94.6, 67.7, 27.8],
    "pcb3":       [83.3, 95.8, 77.0, 42.5],
    "pcb4":       [87.6, 96.1, 84.6, 31.9],
    "pipe_fryum": [98.5, 98.7, 95.6, 55.1],
    "average":    [87.3, 97.2, 84.2, 47.0]
}

MVTEC0 = {
    # iauroc, pauroc, if1, pf1
    "bottle":     [99.2, 89.6, 97.6, 58.1],
    "cable":      [86.5, 77.0, 84.5, 19.7],
    "capsule":    [72.9, 86.9, 91.4, 21.7],
    "carpet":     [100, 95.4, 99.4, 49.7],
    "grid":       [98.8, 82.2, 98.2, 18.6],
    "hazelnut":   [93.9, 94.3, 89.7, 37.6],
    "leather":    [100.0, 96.7, 100, 39.7],
    "metal_nut":  [97.1, 61.0, 96.3, 32.4],
    "pill":       [79.1, 80.0, 91.6, 17.6],
    "screw":      [83.3, 89.6, 87.4, 13.5],
    "tile":       [100, 77.6, 99.4, 32.6],
    "toothbrush": [87.5, 86.9, 87.9, 17.1],
    "transistor": [88.0, 74.7, 79.5, 30.5],
    "wood":       [99.4, 93.4, 98.3, 51.5],
    "zipper":     [91.5, 91.6, 92.9, 34.4],
    "average":    [91.81, 85.12, 92.94, 31.65]
}

MVTEC4 = {
    # iauroc, pauroc, if1, pf1
    "bottle":     [99.3, 97.8, 97.8, 74.3],
    "cable":      [90.9, 94.9, 87.2, 54.7],
    "capsule":    [82.3, 96.2, 92.5, 40.7],
    "carpet":     [100, 99.3,  99.9, 72.0],
    "grid":       [99.6, 98.0, 99.1, 52.7],
    "hazelnut":   [98.4, 98.8, 96.2, 71.0],
    "leather":    [100, 99.3,  99.8, 56.4],
    "metal_nut":  [99.5, 92.9, 98.5, 67.4],
    "pill":       [92.8, 97.1, 94.1, 67.9],
    "screw":      [87.9, 96.0, 89.6, 30.1],
    "tile":       [99.9, 96.6, 99.2, 72.2],
    "toothbrush": [96.7, 98.4, 96.8, 69.0],
    "transistor": [85.7, 88.5, 76.6, 46.6],
    "wood":       [99.8, 95.4, 99.2, 65.1],
    "zipper":     [94.5, 94.2, 94.7, 52.8],
    "average":    [95.2, 96.2, 94.7, 59.5]
}

def make_markdown_table(array):
    nl = "\n"
    markdown = nl
    markdown += f"| {' | '.join(['---']*len(array[0]))} |"
    markdown += nl
    markdown += f"| {' | '.join(array[0])} |"
    markdown += nl
    for entry in array[1:]:
        markdown += f"| {' | '.join(entry)} |{nl}"
    return markdown

def main(args):
    csv_path = args.csv_path
    dataset = args.dataset
    if dataset == "visa0":
        stats_ref = VISA0
    elif dataset == "visa4":
        stats_ref = VISA4
    elif dataset == "mvtec0":
        stats_ref = MVTEC0
    elif dataset == "mvtec4":
        stats_ref = MVTEC4
    else: 
        raise NotImplementedError("no comparision dataset")

    stats = {}
    start_pos, end_pos = -10, -6
    with open(csv_path, "r") as f:
        csvreader = csv.reader(f)
        next(csvreader)
        for row in csvreader:
            if len(row) == 14:
                start_pos, end_pos = -9, -5
            pf1, pauroc, if1, iauroc= row[start_pos: end_pos]
            category = row[-3]
            stats[category] = [iauroc, pauroc, if1, pf1]

    average = [0.0, 0.0, 0.0, 0.0]
    for v in stats.values():
        for i, x in enumerate(v):
            if isinstance(x, str):
                x = float(x)
            average[i] += x
    stats["average"] = [x / len(stats) for x in average]

    for k, v in stats.items():
      stats[k] = [f"{100*float(n):.2f}" for n in v]


    header_array = [["", "i-auroc", "p_auroc", "i-max-f1", "p-max-f1", "i-auroc", "p_auroc", "i-max-f1", "p-max-f1"]]
    stats_array = header_array + [ 
        [k] + [str(n) for n in stats[k]] + [str(n) for n in stats_ref[k]] for k in sorted(stats.keys())
    ]
    markdown_table = make_markdown_table(stats_array)
    with open(csv_path.split("/")[-1][:-3]+"txt", "w") as f:
        f.write(markdown_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument(
        "--dataset", type=str, required=True, 
        help="mvtec0, mvtec4, visa0, visa4, datasetname_<few shot>")
    args = parser.parse_args()
    main(args)