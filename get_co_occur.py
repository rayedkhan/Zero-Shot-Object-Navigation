import openai
import numpy as np

print("here")
openai.api_key = 'xxxxxxx'
print("key verified")
target_objects = [
    "gingerbread house",
    "espresso machine",
    "green plastic crate",
    "white electric guitar",
    "rice cooker",
    "llama wicker basket",
    "whiteboard saying cvpr",
    "tie dye surfboard",
    "blue and red tricycle",
    "graphics card",
    "mate gourd",
    "wooden toy plane",
]

common_objects = ['bed', 'book', 'bottle', 'box', 'knife', 'candle', 'cd', 'cellphone', 'chair', 'cup', 'desk', 'table', 'drawer', 'dresser', 'lamp', 'fork', 'newspaper', 'painting', 'pencil', 'pepper shaker', 'pillow', 'plate', 'pot', 'salt shaker', 'shelf', 'sofa', 'statue', 'tennis racket', 'tv stand', 'watch', 'clock', 'apple', 'baseball bat', 'basketball', 'bowl', 'garbage can', 'plant', 'laptop', 'mug', 'remotecontrol', 'spray bottle', 'television', 'vase', 'wall', "gingerbread house",
    "espresso machine",
    "green plastic crate",
    "white electric guitar",
    "rice cooker",
    "llama wicker basket",
    "whiteboard saying cvpr",
    "tie dye surfboard",
    "blue and red tricycle",
    "graphics card",
    "mate gourd",
    "wooden toy plane"
]

rooms = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']

print("here")
cooccurrence_matrix = np.zeros((len(target_objects), len(common_objects)))
cooccurrence_room_matrix = np.zeros((len(target_objects), len(rooms)))
print("start reasoning")
for i, target in enumerate(target_objects):
    for j, obj in enumerate(common_objects):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"How likely is it to find a {target} and a {obj} together in a normal household setting? Rate from 0 to 1.",
            temperature=0.1,
            max_tokens=10
        )
        print(response)

        probability = float(response.choices[0].text.strip())
        cooccurrence_matrix[i, j] = probability

for i, target in enumerate(target_objects):
    for j, room in enumerate(rooms):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"How likely is it to find a {target} and a {room} together in a normal household setting? Rate from 0 to 1.",
            temperature=0.1,
            max_tokens=10
        )
        print(response)
        
        probability = float(response.choices[0].text.strip())
        cooccurrence_room_matrix[i, j] = probability

np.save("longtail.npy", cooccurrence_matrix) #[12, 55]
np.save("longtail_room.npy", cooccurrence_room_matrix) #[12,9]
