
import openai
import numpy as np
import random
from transformers import ViltProcessor, ViltForQuestionAnswering
from utils_glip import categories_21


'''
sample input:
'''
# frontier_obj_map = [[], [], [], [], [], [], ['box', 'chair', 'table'], ['table', 'painting', 'pot', 'cvpr whiteboard'], ['box', 'chair', 'table'], ['table', 'painting', 'cvpr whiteboard'], ['box', 'chair', 'table'], ['table', 'painting', 'cvpr whiteboard'], ['chair', 'table', 'cvpr whiteboard'], ['table', 'painting', 'cvpr whiteboard'], ['chair', 'table', 'cvpr whiteboard'], ['table', 'cvpr whiteboard'], ['chair', 'table', 'cvpr whiteboard'], ['table', 'cvpr whiteboard'], ['chair', 'table', 'cvpr whiteboard'], ['table', 'cvpr whiteboard'], ['chair', 'table', 'cvpr whiteboard'], ['cvpr whiteboard'], ['chair', 'table', 'cvpr whiteboard'], ['chair', 'table', 'cvpr whiteboard'], ['cvpr whiteboard'], ['cvpr whiteboard'], ['chair', 'table', 'cvpr whiteboard'], ['chair', 'table', 'cvpr whiteboard'], ['cvpr whiteboard'], ['cvpr whiteboard'], ['chair', 'table', 'cvpr whiteboard'], ['chair', 'table', 'cvpr whiteboard'], ['cvpr whiteboard'], ['cvpr whiteboard'], ['chair', 'table', 'cvpr whiteboard'], ['chair', 'table', 'cvpr whiteboard'], ['chair', 'table', 'cvpr whiteboard'], ['chair', 'table', 'cvpr whiteboard'], ['chair', 'table', 'cvpr whiteboard'], ['chair', 'table', 'cvpr whiteboard'], ['chair', 'cvpr whiteboard'], ['cvpr whiteboard'], ['cvpr whiteboard'], ['cvpr whiteboard'], ['cvpr whiteboard'], ['cvpr whiteboard'], ['cvpr whiteboard'], ['cvpr whiteboard']]

# frontier_room_map = [[], [], [], [], [], [], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['bedroom', 'living room', 'dining room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room'], ['living room', 'office room', 'laundry room']]

# goal_prompt = "Alarm Clock"

# openai.api_key = 'YOUR-KEY-HERE'
openai.api_key = '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'


processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

#verification for distractors
def caption_reasoning(goal_prompt, candidate_prompt):
    input_prompt = f"do you think an image with following description {candidate_prompt} would satisfy the condition to occur {goal_prompt}\n\
                    Please answer with yes or no. Note that the spatial information is vital when determining the target since there will be distractors"
    response =  openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # gpt-4
        messages=[{"role": "user", "content": input_prompt}]
        )
    assistant_reply = response.choices[0].message["content"]
    print(input_prompt)
    print("Assistant:", assistant_reply)
    print("yes" in assistant_reply.lower())
    # print("no" in assistant_reply.lower())
    return "yes" in assistant_reply.lower()

#verification for distractors
def image_reasoning(image, prompt):
    encoding = processor(image, prompt, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print(outputs)
    print("Predicted answer:", model.config.id2label[idx])
    return model.config.id2label[idx]

#trimming hidden promptz
def trim_hidden_prompt(hidden_prompt):
    input_prompt = f"we want to find a target object, however the target is hidden and therefore won't be visible for us, so we want to look for a representative object near the target\n\
                    for example, if the target is 'mug under a sofa', we want you to realize that we need to look for sofa instead of a mug.\n\
                    given the folling prompt: {hidden_prompt}, please select a representative object nearby to replace the hidden target. Match your answer exactly letter by letter with {categories_21} and Wrap up your answer with []"
    print(input_prompt)
    response =  openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Not having access to gpt-4 yet
        messages=[{"role": "user", "content": input_prompt}]
    )

    assistant_reply = response.choices[0].message["content"]
    print("Assistant:", assistant_reply)
    start_index = assistant_reply.find("[")
    end_index = assistant_reply.find("]")
    extracted_answer = assistant_reply[start_index + 2:end_index - 1].lower()
    
    print("Extracted Answer: {}".format(extracted_answer))
    # assert extracted_answer in categories_21
    if(extracted_answer not in categories_21):
        second_try = trim_hidden_prompt(hidden_prompt)
        if(second_try in categories_21):
            return second_try
        else:
            return None
    return extracted_answer

#tree of thoughts
def inference(frontier_obj_map, frontier_room_map, goal_prompt):
    trimmed_obj_room_maps = trim_to_median(frontier_obj_map, frontier_room_map)
    input_prompt = "Imagine three different experts are answering this question.\n\
                    They will brainstorm the answer step by step reasoning carefully and taking all facts into consideration\n\
                    All experts will write down 1 step of their thinking, then share it with the group.\n\
                    They will each critique their response, and the all the responses of others They will check their answer based on science and the laws of physics\n\
                    Then all experts will go on to the next step and write down this step of their thinking.\n\
                    They will keep going through steps until they reach their conclusion taking into account the thoughts of the other experts\n\
                    If at any time they realise that there is a flaw in their logic they will backtrack to where that flaw occurred\n\
                    If any expert realises they're wrong at any point then they acknowledges this and start another train of thought\n\
                    Each expert will assign a likelihood of their current assertion being correct\n\
                    Continue until the experts agree on the single most likely location\n\
                    In this task, experts are given a series of locations, along with these locations' nearby informations. They are required to pick one single location where \n\
                    " + goal_prompt + "\n\
                    is most likely to occur, please give a final answer with one single location's number\n\
                    remember to utilize the given spatial information hint and eliminate the locations that the target isn't likely to appear\n\
                    to save time, you can just give out the location number and likelihood for each step"
    for idx in trimmed_obj_room_maps:
        input_prompt += ("\nlocation #" + str(idx) + ": located near " + str(frontier_room_map[idx]) + ", where " + str(frontier_obj_map[idx]) + " also occurs")
    print(input_prompt)
    response =  openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Not having access to gpt-4 yet
        messages=[{"role": "user", "content": input_prompt}]
    )

    assistant_reply = response.choices[0].message["content"]
    print("Assistant:", assistant_reply)
    goal_idx = extract_last_integer(assistant_reply)
    print("goal idx: ", goal_idx)
    if(goal_idx not in trimmed_obj_room_maps):
        i = random.randint(0, len(trimmed_obj_room_maps))
        return trimmed_obj_room_maps[i]
    return goal_idx

def extract_last_integer(text):
    words = text.split()  # Split the text into words
    for word in reversed(words):
        if word.lstrip("#").isdigit():  # Check if the word is a numeric string after removing any leading "#"
            return int(word.lstrip("#"))  # Convert and return the integer
    return None  # Return None if no integer is found




#trimming out consequent locations that shares the same information
def trim_to_median(frontier_obj_map, frontier_room_map):
    idx = 0
    assert len(frontier_obj_map) == len(frontier_room_map)
    trimmed_obj_room_maps = []
    consequent_room_maps = []
    consequent_obj_maps = []
    while(idx < len(frontier_obj_map)):
        curr_obj_map = frontier_obj_map[idx]
        curr_room_map = frontier_room_map[idx]
        if(curr_obj_map == [] and curr_room_map == []):
            idx+=1
            continue

        if(len(consequent_room_maps) == 0 and len(consequent_obj_maps) == 0):   #the first loop
            consequent_room_maps.append((idx, curr_room_map))
            consequent_obj_maps.append((idx, curr_obj_map))
            idx+=1
            continue

        if(curr_obj_map == consequent_obj_maps[-1][1] and curr_room_map == consequent_room_maps[-1][1]):
            consequent_room_maps.append((idx, curr_room_map))
            consequent_obj_maps.append((idx, curr_obj_map))
            idx+=1
            continue
        else:    #the previous map is different from the current map
            assert len(consequent_room_maps) == len(consequent_obj_maps)
            middle_idx = consequent_room_maps[len(consequent_obj_maps) // 2][0]
            trimmed_obj_room_maps.append(middle_idx)
            consequent_room_maps = []
            consequent_obj_maps = []
    if(len(consequent_room_maps) != 0 and len(consequent_obj_maps) != 0):
        middle_idx = consequent_room_maps[len(consequent_obj_maps) // 2][0]
        trimmed_obj_room_maps.append(middle_idx)
        consequent_room_maps = []
        consequent_obj_maps = []
    return trimmed_obj_room_maps



