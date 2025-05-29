import csv
import gzip
import json


# GLIP prompt
categories = [] # categories except doors 
categories_40 = []
categories_map = {}
categories_doors = []

# ['bed', 'book', 'bottle', 'box', 'knife', 'candle', 'cd', 'cellphone', 'chair', 'cup', 'desk', 'table', 'drawer', 'dresser', 'lamp', 'fork', 'newspaper', 'painting', 'pencil', 'pepper shaker', 'pillow', 'plate', 'pot', 'salt shaker', 'shelf', 'sofa', 'statue', 'tennis racket', 'tv stand', 'watch', 'clock', 'apple', 'baseball bat', 'basketball', 'bowl', 'garbage can', 'plant', 'laptop', 'mug', 'remotecontrol', 'spray bottle', 'television', 'vase', 'wall']
categories_21 = [
    "Bed",
    "Book",
    "Bottle",
    "Box",
    "Knife",
    "Candle",
    "CD",
    "CellPhone",
    "Chair",
    "Cup",
    #"DeskLamp",
    "Desk",
    "Table",
    "Drawer",
    "Dresser",
    "Lamp",
    "Fork",
    "Newspaper",
    "Painting",
    "Pencil",
    #"Pen",
    "Pepper Shaker",
    "Pillow",
    "Plate",
    "Pot",
    "Salt Shaker",
    "Shelf",
    "Sofa",
    "Statue",
    # "Teddy Bear",
    "Tennis Racket",
    "TV Stand",
    "Watch",

    "Remotecontrol",
    "Clock",
    "Apple",
    "Baseball bat",
    "BasketBall",
    "Bowl",
    "garbage can",
    "Plant",
    "Laptop",
    "Mug",
    "Spray Bottle",
    "Television",
    "Vase",
    ]
categories_21 = [obj.lower() for obj in categories_21]



    
goal_objects = ["Clock",
    "Apple",
    "Baseball bat",
    "BasketBall",
    "Bowl",
    "garbage can",
    "Plant",
    "Laptop",
    "Mug",
    "Remotecontrol",
    "Spray Bottle",
    "Television",
    "Vase"]
goal_objects = [obj.lower() for obj in goal_objects]

rooms = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']
rooms_captions = '. '.join(rooms)+'.'

door_captions = 'doorway. hallway.'# v2
object_captions = '. '.join(categories_21)+'.' # + '. wall. door.'

# pre_defined_captions = rooms + pre_defined_captions


# LLM reasoning prompt
# room_prompt = "In which room will you most likely to find a "

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    bb1 = {'x1': bb1[0], 'x2': bb1[2], 'y1': bb1[1], 'y2': bb1[3]}
    bb2 = {'x1': bb2[0], 'x2': bb2[2], 'y1': bb2[1], 'y2': bb2[3]}
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
