# This file is under the MIT License and not bounded by Gaussian Splatting License
#MIT License
#
#Copyright (c) 2024 Tomás Berriel Martins
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  copies of the Software, and to permit persons to whom Christopher Johannes Wewerthe Software is furnished to do so, subject to the following conditions:
# 
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import csv
import torch

def loadSemanticClasses(semantic_classes_path, n = 32, load_file = True):
    data = []
    weights = []
    
    with open(semantic_classes_path, "r") as file:
        lines = csv.DictReader(file)
        for row in lines:
            row["idx"] = int(row["idx"])
            row["rgb"] = [float(row["r"]), float(row["g"]), float(row["b"])]
            row["count"] = int(row["count"])
            
            data.append(row) 

            weights.append(1/row["count"])
        weights = np.array(weights)*100
        weights[-1] = weights[:-1].min()
        
    return data, torch.tensor(weights).float()

def mapClassesToRGB(seg_image, map_data):
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
                legend_classes["labels"].append(map_data[i]["class"])
                legend_classes["rgb"].append(map_data[i]["rgb"])
                rgb_image = torch.where(seg_image == i, torch.tensor(map_data[i]["rgb"],device=device), rgb_image)
    
    return rgb_image.cpu().numpy(), legend_classes

