import csv
import torch

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import os

def loadClassesMapping(file_name: str ="/media/tberriel/My_Book_2/ScanNetpp/data/metadata/semantic_benchmark/map_benchmark.csv"):
    """
    Reads data from a CSV file and stores it in a list ordered by the "idx" column.

    Args:
        file_name (str): Path to the CSV file.

    Returns:
        list: List of rows, where each row is a dictionary containing "idx", "label", "red", "green", and "blue".
    """

    n = 1651
    colours = colors.ListedColormap(plt.cm.tab20b.colors + plt.cm.tab20c.colors , name="tab20_extended")
    cmap = colours(np.linspace(0, 1, n))  # Obtain RGB colour map
    #cmap[0,-1] = 0  # Set alpha for label 0 to be 0
    #cmap[1:,-1] = 0.3  # Set the other alphas for the labels to be 0.3

    data = []
    with open(file_name, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for i,row in enumerate(reader):
            # Convert values to appropriate data types
            row["idx"] = int(row[""])
            row["class"] = str(row["class"])
            row["count"] = int(row["count"])
            data.append(row)

    
    return data

def getTopClassesMapping(file_name: str ="/media/tberriel/My_Book_2/ScanNetpp/decoded_data_nerfstudio/32_most_common_classes.csv", n=32):
    """
    Reads data from a CSV file and stores it in a list ordered by the "idx" column.

    Args:
        file_name (str): Path to the CSV file.

    Returns:
        list: List of rows, where each row is a dictionary containing "idx", "label", "red", "green", and "blue".
    """
    data = loadClassesMapping()
    data.sort(key=lambda row:row["count"])
    data =  data[-n:] if n > 0 else data
    data_dict = {"idx":[],"class":[],"count":[]}
    for line in data:
        data_dict["idx"].append(line["idx"])
        data_dict["class"].append(line["class"])
        data_dict["count"].append(line["count"])
    return data_dict

def loadSemanticClasses(n = 63, load_file = True):
    data = []
    map_file ="/media/tberriel/My_Book_2/ScanNetpp/decoded_data_nerfstudio/"+str(n)+"_most_common_classes.csv"
    if os.path.exists(map_file) and load_file:
        with open(map_file, "r") as file:
            lines = csv.DictReader(file)
            for row in lines:
                row["idx"] = int(row["idx"])
                row["rgb"]=[float(row["r"]), float(row["g"]), float(row["b"])]
                
                data.append(row) 
    else:
        file_sem_clases ="/media/tberriel/My_Book_2/ScanNetpp/data/metadata/semantic_classes.txt"
        colours = colors.ListedColormap(plt.cm.tab20b.colors + plt.cm.tab20c.colors , name="tab20_extended")
        cmap = colours(np.linspace(0, 1, n))  # Obtain RGB colour map
        top_classes = getTopClassesMapping(n=n)
        i_col = 0
        with open(file_sem_clases, "r") as file:
            lines = file.readlines()
            for i,line in enumerate(lines):
                if line.strip() in top_classes["class"]:
                    # Convert values to appropriate data types
                    row = dict()
                    row["idx"] = i
                    row["class"] = line.strip()
                    row["rgb"]=cmap[i_col,:3]
                    i_col+=1

                    # Store row in the list
                    data.append(row) 
        if True:
            file_path = "/media/tberriel/My_Book_2/ScanNetpp/decoded_data_nerfstudio/"+str(n)+"_most_common_classes.csv"
            new_data = []
            for row in data:
                row["r"] = row["rgb"][0]
                row["g"] = row["rgb"][1]
                row["b"] = row["rgb"][2]
                row.pop("rgb")
            with open(file_path, "w", newline="") as csvfile:
                fieldnames = data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)   
    return data

def mapClassesToRGB(seg_image, map_data, empty_value=65535):
    """
    Maps each pixel in an image with idx values to corresponding RGB colors from data.

    Args:
        seg_image (np.array): The 2D image with idx values for each pixel.
        map_data (list): The list of color information from the CSV file.

    Returns:
        PIL.Image: The image with pixels converted to RGB colors.
    """
    device = seg_image.device
    # Create a new RGB image
    rgb_image = torch.zeros(list(seg_image.shape)+[3],device=device)
    seg_image = seg_image.unsqueeze(-1)

    # Map each pixel value to its color
    legend_classes = {"labels":[],"rgb":[]}
    for i in range(seg_image.max()+1):
        if (seg_image==i).sum()>0:
            if i >= len(map_data):
                rgb_image = torch.where(seg_image == seg_image.max(), torch.tensor([1,1,1],device=device), rgb_image)
            else:
                legend_classes["labels"].append(map_data[i]["class"])
                legend_classes["rgb"].append(map_data[i]["rgb"])
            rgb_image = torch.where(seg_image == i, torch.tensor(map_data[i]["rgb"],device=device), rgb_image)
    
    return rgb_image.cpu().numpy(), legend_classes

