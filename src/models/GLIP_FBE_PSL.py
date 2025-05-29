import argparse
import imp
from multiprocessing.context import ForkContext
import os
import math
import numba
import time
import random
import string
import numpy as np
import skimage
import torch
import pandas
from torchvision.utils import save_image
import copy
from PIL import Image
from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete
import matplotlib.pyplot as plt
from matplotlib import colors

from GLIP.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from GLIP.maskrcnn_benchmark.config import cfg as glip_cfg
from utils_glip import *

from utils_fmm.fmm_planner import FMMPlanner
from utils_fmm.mapping import Semantic_Mapping
import utils_fmm.control_helper as CH
import utils_fmm.pose_utils as pu
from src.models.agent import Agent
from robothor_challenge import ALLOWED_ACTIONS

from pslpython.model import Model as PSLModel
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

from src.simulation.constants import THOR_LONGTAIL_OBJECT_TYPES_CLIP, THOR_LONGTAIL_TYPES


ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres'
]

object_captions += ' wall.'
obj_mapping = {'alarmclock':'clock', 'apple':'apple', 'baseballbat': 'baseball bat', 'basketball': 'basketball', 'bowl': 'bowl', 'garbagecan': 'garbage can', 'houseplant': 'plant', 'laptop': 'laptop', 'mug': 'mug', 'remotecontrol': 'remotecontrol', 'spraybottle': 'spray bottle', 'television': 'television', 'vase': 'vase'}


def add_longtail():
    global categories_21
    categories_21 += [  "gingerbread house",
                        "espresso machine",
                        "green plastic crate",
                        "white electric guitar",
                        "rice cooker",
                        "llama wicker basket",
                        # "whiteboard saying cvpr",
                        "cvpr whiteboard",
                        "tie dye surfboard",
                        # "blue and red tricycle",
                        "blue red tricycle",
                        "graphics card",
                        "mate gourd",
                        "wooden toy plane"]

    global object_captions
    object_captions = '. '.join(categories_21)+'.' # + '. wall. door.'


def add_remap(values):
    assert len(values) == 12
    values = list(values)
    for i in range(12):
        values[i] = values[i].replace(",", "")
    print(values)
    global categories_21
    categories_21[-12:] = values

    global object_captions
    object_captions = object_captions = '. '.join(categories_21)+'.' # + '. wall. door.'

class CLIP_LLM_FMMAgent_NonPano(Agent):
    """
    New in this version: 
    1. use obj and room reasoning by record object locations and build a room map 
    experiments: v4_4
    """
    def __init__(self, args=None):

        self.EnvTypes = str(args.EnvTypes)
        self.ClassTypes = str(args.ClassTypes)

        self.obj_mapping = obj_mapping

        self.args = args
        self.panoramic = []
        self.panoramic_depth = []
        self.turn_angles = 0
        self.device = (
            torch.device("cuda:{}".format(0))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.prev_action = 0
        self.current_search_room = ''
        self.navigate_steps = 0
        self.move_steps = 0
        self.total_steps = 0
        self.found_goal = False
        self.correct_room = False
        self.changing_room = False
        self.changing_room_steps = 0
        self.move_after_new_goal = False
        self.former_check_step = -10
        self.goal_disappear_step = 100
        self.force_change_room = False
        self.current_room_search_step = 0
        self.target_room = ''
        self.current_rooms = []
        self.nav_without_goal_step = 0
        self.former_collide = 0
        self.agent_gps = np.array([0.,0.])

        # init glip model
        config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml"
        weight_file = "GLIP/MODEL/glip_large_model.pth"
        # config_file = "GLIP/configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
        # weight_file = "GLIP/MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"
        glip_cfg.local_rank = 0
        glip_cfg.num_gpus = 1
        glip_cfg.merge_from_file(config_file)
        glip_cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        glip_cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        self.glip_demo = GLIPDemo(
            glip_cfg,
            min_image_size=800,
            confidence_threshold=0.61,
            show_mask_heatmaps=False
        )
        print('glip init finish!!!')
        self.map_size_cm = 3000
        self.resolution = self.map_resolution = 5
        self.camera_horizon = 30
        self.dilation_deg = 0
        self.collision_threshold = 0.08
        self.col_width = 5
        self.selem = skimage.morphology.square(1)
        self.init_map()
        self.sem_map_module = Semantic_Mapping(self).to(self.device) 
        self.free_map_module = Semantic_Mapping(self, max_height=10,min_height=-150).to(self.device)
        self.room_map_module = Semantic_Mapping(self, max_height=150,min_height=-10, num_cats=9).to(self.device)
        
        self.free_map_module.eval()
        self.free_map_module.set_view_angles(self.camera_horizon)
        self.sem_map_module.eval()
        self.sem_map_module.set_view_angles(self.camera_horizon)
        self.room_map_module.eval()
        self.room_map_module.set_view_angles(self.camera_horizon)
        
        print('FMM navigate init finish!!!')
        
        self._POSSIBLE_ACTIONS = ["MoveAhead", "RotateRight", "RotateLeft", "LookUp", "LookDown", "Stop"]
        self.action_projection = {0:"Stop", 1:"MoveAhead", 2:"RotateLeft", 3:"RotateRight", 4:"LookUp", 5:"LookDown"}
        
        self.goal_idx = {}
        self.goals = goal_objects
        if self.args.llm == 'deberta':
            
            for goal in self.goals:                
                self.goal_idx[goal] = self.goals.index(goal) # each goal corresponding to which row in co-orrcurance matrix 
        
            # self.co_occur_mtx = np.load('ablations/npys/deberta_predict.npy')
            if(self.EnvTypes=="EnvTypes.LONGTAIL"):
                self.co_occur_mtx = np.load('ablations/npys/longtail.npy')
                self.co_occur_mtx -= self.co_occur_mtx.min()
                self.co_occur_mtx /= self.co_occur_mtx.max()
                
                # self.co_occur_room_mtx = np.load('ablations/npys/deberta_predict_room.npy')
                
                self.co_occur_room_mtx = np.load('ablations/npys/longtail_room.npy')
                self.co_occur_room_mtx -= self.co_occur_room_mtx.min()
                self.co_occur_room_mtx /= self.co_occur_room_mtx.max()
            else:
                self.co_occur_mtx = np.load('ablations/npys/deberta_predict.npy')
                self.co_occur_mtx -= self.co_occur_mtx.min()
                self.co_occur_mtx /= self.co_occur_mtx.max()
                
                self.co_occur_room_mtx = np.load('ablations/npys/deberta_predict_room.npy')
                self.co_occur_room_mtx -= self.co_occur_room_mtx.min()
                self.co_occur_room_mtx /= self.co_occur_room_mtx.max()
                
            if self.args.PSL_infer == 'optim':
                self.psl_model = PSLModel('objnav3')
                # Add Predicates
                self.add_predicates(self.psl_model)

                # Add Rules
                self.add_rules(self.psl_model)
        else:
            self.goals = ['chair','sofa','plant','bed','toilet','tv_monitor']
            for goal in self.goals :
                self.goal_idx[goal] = self.goals.index(goal) # each goal corresponding to which row in co-orrcurance matrix 
            self.co_occur_mtx = np.load('ablations/npys/chatgpt_obj.npy')
            self.co_occur_room_mtx = np.load('ablations/npys/chatgpt_room.npy')
        
        


    def remove_punctuation(self, text):
        # Create a translation table to map punctuation to None
        translator = str.maketrans('', '', string.punctuation)
        # Apply the translation to remove punctuation
        text = text.translate(translator)
        return text
    
    def remap_classes(self, remap):
        keys = remap.keys()
        values = remap.values()
        values = list(values)
        for i, v in enumerate(values):
            values[i] = self.remove_punctuation(v.lower())
        for i, k in enumerate(keys):
            for j, v in enumerate(values):
                if(i == j):
                    self.obj_mapping[k.lower()] = v
                    break
        
        self.goal_idx = {}
        for i, k in enumerate(self.obj_mapping):
            self.goal_idx[self.obj_mapping[k]] = i
        add_remap(values)
        print("bject_captions: ", object_captions)
        print("self.goal_idx: ", self.goal_idx)
        print("self.obj_mapping: ", self.obj_mapping)
        # exit()
        
    def remap_longtail(self):
        self.obj_mapping = {}
        
        for i in range(0, len(THOR_LONGTAIL_TYPES)):
            key = THOR_LONGTAIL_TYPES[i].lower()
            val = THOR_LONGTAIL_OBJECT_TYPES_CLIP[i].lower()
            self.obj_mapping[key] = val
        self.mode = 1
        add_longtail()
        self.goal_idx = {}
        for i in range(0, len(THOR_LONGTAIL_OBJECT_TYPES_CLIP)):
            self.goal_idx[THOR_LONGTAIL_OBJECT_TYPES_CLIP[i].lower()] = i
        print(self.goal_idx)
    
    def add_predicates(self, model):
        if self.args.reasoning in ['both', 'obj']:

            predicate = Predicate('IsNearObj', closed = True, size = 2)
            model.add_predicate(predicate)
            
            predicate = Predicate('ObjCooccur', closed = True, size = 1)
            model.add_predicate(predicate)
        if self.args.reasoning in ['both', 'room']:

            predicate = Predicate('IsNearRoom', closed = True, size = 2)
            model.add_predicate(predicate)
            
            predicate = Predicate('RoomCooccur', closed = True, size = 1)
            model.add_predicate(predicate)
        
        predicate = Predicate('Choose', closed = False, size = 1)
        model.add_predicate(predicate)
        
        predicate = Predicate('ShortDist', closed = True, size = 1)
        model.add_predicate(predicate)
        
    def add_rules(self, model):
        if self.args.reasoning in ['both', 'obj']:
            model.add_rule(Rule('1: ObjCooccur(O) & IsNearObj(O,F)  -> Choose(F)^2'))
            model.add_rule(Rule('1: !ObjCooccur(O) & IsNearObj(O,F) -> !Choose(F)^2'))
        if self.args.reasoning in ['both', 'room']:
            model.add_rule(Rule('1: RoomCooccur(R) & IsNearRoom(R,F) -> Choose(F)^2'))
            model.add_rule(Rule('1: !RoomCooccur(R) & IsNearRoom(R,F) -> !Choose(F)^2'))
        model.add_rule(Rule('1: ShortDist(F) -> Choose(F)^2'))
        model.add_rule(Rule('Choose(+F) = 1 .'))

    
    def reset(self):
        self.navigate_steps = 0
        self.turn_angles = 0
        self.move_steps = 0
        self.total_steps = 0
        self.current_room_search_step = 0
        self.found_goal = False
        self.correct_room = False
        self.changing_room = False
        self.changing_room_steps = 0
        self.move_after_new_goal = False
        self.former_check_step = -10
        self.goal_disappear_step = 100
        self.prev_action = 0
        self.fake_prev_action = 0
        self.col_width = 5
        self.former_collide = 0
        self.goal_gps = np.array([110.,110.])
        self.agent_gps = np.array([0.,0.])
        self.last_gps = np.array([11100.,11100.])
        self.turn_angle = np.array([0.])
        
        self.has_panarama = False
        self.init_map()
        self.sem_map_module.set_view_angles(30)
        self.free_map_module.set_view_angles(30)
        self.last_loc = self.full_pose
        self.panoramic = []
        self.panoramic_depth = []
        self.current_rooms = []
        self.dist_to_frontier_goal = 10
        self.first_fbe = False
        self.goal_map = np.zeros(self.full_map.shape[-2:])
        self.last_img = None
        ###########
        self.current_obj_predictions = []
        self.obj_locations = [[] for i in range(len(categories_21))] # length equal to all the objects in reference matrix 
        self.not_move_steps = 0
        self.move_since_random = 0
        self.using_random_goal = False
        
        self.fronter_this_ex = 0
        self.random_this_ex = 0
        ########### error analysis
        self.detect_true = False
        self.goal_appear = False
        self.frontiers_gps = []
        
        self.last_location = np.array([0.,0.])
        self.current_stuck_steps = 0
        self.total_stuck_steps = 0
    
    def not_move(self, depth):
        """
        judge if the agent moves or not using depth info
        """
        if np.sum((self.last_img-depth) <= 0.05) > 0.9 * depth.shape[0]*depth.shape[1]:
            return True
        else:
            return False
        
        
    def detect_objects(self, observations):
        # print("object_captions: " , object_captions) #object_captions = 43 common objects + "wall"
        self.current_obj_predictions = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], object_captions) # self.glip_panorama(object_captions) # time cosuming
        # print(len(self.glip_demo.entities), self.glip_demo.entities)
        # print("current_obj_predictions: ", self.current_obj_predictions)    # current_obj_predictions = BoxList(num_boxes=1, image_width=640, image_height=480, mode=xyxy)
        new_labels = self.get_glip_real_label(self.current_obj_predictions)
        print("new object labels: ", new_labels)  #real labels: ['bowl', 'pencil', 'fork', 'object']
        self.current_obj_predictions.add_field("labels", new_labels)
        shortest_distance = 10 # TODO: shortest distance or most confident?
        shortest_distance_angle = 0
        goal_prediction = copy.deepcopy(self.current_obj_predictions)
        obj_labels = self.current_obj_predictions.get_field("labels")
        goal_bbox = []
        for j, label in enumerate(obj_labels):
            if self.obj_goal == label:
                goal_bbox.append(self.current_obj_predictions.bbox[j])
        '''map the object location'''
        if self.args.reasoning in ['both', 'obj']:
            for j, label in enumerate(obj_labels):
                if label in categories_21:
                    confidence = self.current_obj_predictions.get_field("scores")[j]
                    bbox = self.current_obj_predictions.bbox[j].to(torch.int64)
                    center_point = (bbox[:2] + bbox[2:]) // 2
                    temp_direction = (center_point[0] - 320) * 79 / 640
                    temp_distance = self.depth[center_point[1],center_point[0]]
                    obj_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                    x = int(self.map_size_cm/10-obj_gps[1]*100/self.resolution)
                    y = int(self.map_size_cm/10+obj_gps[0]*100/self.resolution)
                    self.obj_locations[categories_21.index(label)].append([confidence, x, y])
                
        if len(goal_bbox) > 0:
            goal_prediction.bbox = torch.stack(goal_bbox)
            for box in goal_prediction.bbox:
                box = box.to(torch.int64)
                center_point = (box[:2] + box[2:]) // 2
                temp_direction = (center_point[0] - 320) * 79 / 640
                temp_distance = self.depth[center_point[1],center_point[0]]
                k = 0
                pos_neg = 1
                while temp_distance <= 0 and 0<center_point[1]+int(pos_neg*k)<480 and 0<center_point[0]+int(pos_neg*k)<640:
                    pos_neg *= -1
                    k += 0.5
                    temp_distance = max(self.depth[center_point[1]+int(pos_neg*k),center_point[0]],
                    self.depth[center_point[1],center_point[0]+int(pos_neg*k)])
                new_goal_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                self.found_goal = True
                    
                direction = temp_direction
                distance = temp_distance
                if distance < shortest_distance:
                    shortest_distance = distance
                    shortest_distance_angle = direction
                    if self.args.error_analysis:
                        box_shortest = copy.deepcopy(box)
            
            self.goal_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            if self.args.error_analysis:
                if (observations['semantic'][box_shortest[0]:box_shortest[2],box_shortest[1]:box_shortest[3]] == self.goal_mp3d_idx).sum() > min(300, 0.2 * (box_shortest[2]-box_shortest[0])*(box_shortest[3]-box_shortest[1])):
                     self.detect_true = True
            
                   
    def act(self, observations):
        """ 
        observations: 'object_goal' = "AlarmClock", "depth" = (480, 640), "rgb"=(480, 640, 3)
        """ 

        if self.args.error_analysis:
            if np.linalg.norm(observations['gps'] - self.last_location) < 1:
                if self.current_stuck_steps == 50:
                    self.total_stuck_steps += 50
                if self.current_stuck_steps > 50:
                    self.total_stuck_steps += 1
                self.current_stuck_steps += 1
            else:
                self.current_stuck_steps = 0
                self.last_location = observations['gps']
            
        if self.total_steps >= 499:
            return self.action_projection[0]
        self.total_steps += 1
        
        if self.navigate_steps == 0:
            ## initialize commonsense 
            self.obj_goal = self.obj_mapping[observations["object_goal"].lower()]
            self.prob_array_room = self.co_occur_room_mtx[self.goal_idx[self.obj_goal]] 
            self.prob_array_obj = self.co_occur_mtx[self.goal_idx[self.obj_goal]]   
            print("looking for ", self.obj_goal)

            #default to one-hot, not optim -- not entering this if-statement
            if self.args.PSL_infer == 'optim':              
                if self.args.reasoning in ['both','room']:
                    for predicate in self.psl_model.get_predicates().values():
                        if predicate.name() in ['ROOMCOOCCUR']:
                            predicate.clear_data()
                    prob_array_room_list = list(self.prob_array_room)
                    data = pandas.DataFrame([[i, prob_array_room_list[i]] for i in range(len(prob_array_room_list))], columns = list(range(2)))
                    self.psl_model.get_predicate('RoomCooccur').add_data(Partition.OBSERVATIONS, data)
                
                if self.args.reasoning in ['both','obj']:
                    for predicate in self.psl_model.get_predicates().values():
                        if predicate.name() in ['OBJCOOCCUR']:
                            predicate.clear_data()
                    prob_array_obj_list = list(self.prob_array_obj)
                    data = pandas.DataFrame([[i, prob_array_obj_list[i]] for i in range(len(prob_array_obj_list))], columns = list(range(2)))
                    self.psl_model.get_predicate('ObjCooccur').add_data(Partition.OBSERVATIONS, data)

            # if self.args.error_analysis:
            #     self.goal_mp3d_idx = mp3d_cat_to_idx[self.obj_goal]
                
        if self.args.error_analysis:
            if (observations['semantic'] == self.goal_mp3d_idx).sum() > 300:
                self.goal_appear = True
        
        self.depth = observations["depth"]
        self.rgb = observations["rgb"][:,:,[2,1,0]]
        
        # judge if not move and update gps
        #update the angle if the previous action caused a turn
        if self.fake_prev_action == 6:
            self.turn_angle -= np.pi / 3
        #rotate left
        elif self.fake_prev_action == 2:
            self.turn_angle += np.pi / 6
        #rotate right
        elif self.fake_prev_action == 3:
            self.turn_angle -= np.pi / 6
        #if moving ahead, update the robot's location in x and y axis
        elif self.fake_prev_action == 1:
            if not self.not_move(observations['depth']):    #judge if the agent moves or not using depth info
                self.agent_gps[0] += 0.25*np.cos(self.turn_angle)
                self.agent_gps[1] -= 0.25*np.sin(self.turn_angle)
                
        observations['compass'] = self.turn_angle
        observations['gps'] = self.agent_gps
        self.last_img = observations['depth']

        self.update_map(observations)
        self.update_free_map(observations)

        # initialize map at the begining, look around first
        if self.total_steps <= 6:
            self.fake_prev_action = 3
            return self.action_projection[3]
        elif self.total_steps <= 7:
            self.fake_prev_action = 3
            self.sem_map_module.set_view_angles(0)
            self.free_map_module.set_view_angles(0)
            self.room_map_module.set_view_angles(0)
            return self.action_projection[3]
        if self.total_steps <= 13 and not self.found_goal:
            self.panoramic.append(observations["rgb"][:,:,[2,1,0]])
            self.panoramic_depth.append(observations["depth"])
            self.detect_objects(observations)
            if self.args.reasoning in ['both','room']:
                # print("rooms_captions: ", rooms_captions) #bedroom. living room. bathroom. kitchen. dining room. office room. gym. lounge. laundry room.
                room_detection_result = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], rooms_captions)
                self.update_room_map(observations, room_detection_result)
            if not self.found_goal:
                self.fake_prev_action = 3
                return self.action_projection[3]
        
        #update the move_steps and not_move_steps based on if the agent has changed its location
        if not (observations["gps"] == self.last_gps).all():
            self.move_steps += 1
            self.not_move_steps = 0
            if self.using_random_goal:
                self.move_since_random += 1
        else:
            self.not_move_steps += 1
            
        self.last_gps = copy.deepcopy(observations["gps"])
        
        if not self.found_goal:
            ## detect objects and rooms if have not detected a goal.
            self.detect_objects(observations)
            if self.total_steps % 2 == 0 and self.args.reasoning in ['both','room']:
                room_detection_result = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], rooms_captions)
                self.update_room_map(observations, room_detection_result)
        
        ### ----- generate action using FMM ----- ###

        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose.cpu().numpy()   #last location
        input_pose[1] = self.map_size_cm/100 - input_pose[1]
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]
        traversible, cur_start, cur_start_o = self.get_traversible(self.full_map.cpu().numpy()[0,0,::-1], input_pose)
        
        if self.found_goal: ## directly go to goal
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            self.goal_map[max(0,min(599,int(self.map_size_cm/10+self.goal_gps[1]*100/self.resolution))), max(0,min(599,int(self.map_size_cm/10+self.goal_gps[0]*100/self.resolution)))] = 1
        elif not self.first_fbe:  ## first FBE process
            self.goal_loc = self.fbe(traversible, cur_start)
            self.not_use_random_goal()
            self.first_fbe = True
            if self.goal_loc is None:
                self.goal_map = self.set_random_goal()
                self.using_random_goal = True
            else:
                self.goal_map[self.goal_loc[0], self.goal_loc[1]] = 1
                self.goal_map = self.goal_map[::-1]
                if self.args.error_analysis:
                    self.frontiers_gps.append([(self.goal_loc[1]-self.map_size_cm/10) * self.resolution / 100, (self.map_size_cm/10 - self.goal_loc[0]) * self.resolution / 100])
        
        stg_y, stg_x, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        if (not self.found_goal and number_action == 0) or (self.using_random_goal and self.move_since_random > 20):
            self.goal_loc = self.fbe(traversible, cur_start)
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            if self.goal_loc is None:
                self.goal_map = self.set_random_goal()
                self.using_random_goal = True
            else:
                self.goal_map[self.goal_loc[0], self.goal_loc[1]] = 1
                self.goal_map = self.goal_map[::-1]
                if self.args.error_analysis:
                    self.frontiers_gps.append([(self.goal_loc[1]-self.map_size_cm/10) * self.resolution / 100, (self.map_size_cm/10 - self.goal_loc[0]) * self.resolution / 100])
        
            stg_y, stg_x, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        
        loop_time = 0
        while (not self.found_goal and number_action == 0) or self.not_move_steps >= 7: 
            ## if the agent is stuck, then randomly select a goal
            loop_time += 1
            if loop_time > 20:
                return self.action_projection[0]
            self.not_move_steps = 0
            self.goal_map = self.set_random_goal()
            self.using_random_goal = True
            stg_y, stg_x, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        # ------------------------------
        if self.args.visulize:
            # print("==========visualizing==========")
            save_map = copy.deepcopy(torch.from_numpy(traversible))
            gray_map = torch.stack((save_map, save_map, save_map))
            paper_obstacle_map = copy.deepcopy(gray_map)[:,1:-1,1:-1]
            gray_map[:, int((self.map_size_cm/100-self.full_pose[1])*100/self.resolution)-2:int((self.map_size_cm/100-self.full_pose[1])*100/self.resolution)+2, int(self.full_pose[0]*100/self.resolution)-2:int(self.full_pose[0]*100/self.resolution)+2] = 0
            gray_map[0, int((self.map_size_cm/100-self.full_pose[1])*100/self.resolution)-2:int((self.map_size_cm/100-self.full_pose[1])*100/self.resolution)+2, int(self.full_pose[0]*100/self.resolution)-2:int(self.full_pose[0]*100/self.resolution)+2] = 1
            if not self.found_goal and self.goal_loc is not None:
                gray_map[:,int(self.map_size_cm/5)-self.goal_loc[0]-2:int(self.map_size_cm/5)-self.goal_loc[0]+2, self.goal_loc[1]-2:self.goal_loc[1]+2] = 0
                gray_map[1,int(self.map_size_cm/5)-self.goal_loc[0]-2:int(self.map_size_cm/5)-self.goal_loc[0]+2, self.goal_loc[1]-2:self.goal_loc[1]+2] = 1
            else:
                gray_map[:, int((self.map_size_cm/200+self.goal_gps[1])*100/self.resolution)-2:int((self.map_size_cm/200+self.goal_gps[1])*100/self.resolution)+2, int((self.map_size_cm/200+self.goal_gps[0])*100/self.resolution)-2:int((self.map_size_cm/200+self.goal_gps[0])*100/self.resolution)+2] = 0
                gray_map[1, int((self.map_size_cm/200+self.goal_gps[1])*100/self.resolution)-2:int((self.map_size_cm/200+self.goal_gps[1])*100/self.resolution)+2, int((self.map_size_cm/200+self.goal_gps[0])*100/self.resolution)-2:int((self.map_size_cm/200+self.goal_gps[0])*100/self.resolution)+2] = 1
            gray_map[:, int(stg_y)-2:int(stg_y)+2, int(stg_x)-2:int(stg_x)+2] = 0
            gray_map[2, int(stg_y)-2:int(stg_y)+2, int(stg_x)-2:int(stg_x)+2] = 1
            free_map = self.fbe_free_map.cpu().numpy()[0,0,::-1].copy() > 0.5
            
            paper_map = torch.zeros_like(paper_obstacle_map)
            paper_map_trans = paper_map.permute(1,2,0)
            unknown_rgb = colors.to_rgb('lightcyan')
            paper_map_trans[:,:,:] = torch.tensor( unknown_rgb)
            free_rgb = colors.to_rgb('floralwhite')
            paper_map_trans[self.fbe_free_map.cpu().numpy()[0,0,::-1]>0.5,:] = torch.tensor( free_rgb).double()
            frontier_rgb = colors.to_rgb('indianred')
            selem = skimage.morphology.disk(1)
            free_map[skimage.morphology.binary_dilation(free_map, selem)] = 1
            paper_map_trans[(free_map==1)*(paper_map_trans[:,:,0]==torch.tensor(unknown_rgb)[0]).numpy(),:] = torch.tensor(frontier_rgb).double()
            obstacle_rgb = colors.to_rgb('dimgrey')
            paper_map_trans[skimage.morphology.binary_dilation(self.full_map.cpu().numpy()[0,0,::-1]>0.5,skimage.morphology.disk(1)),:] = torch.tensor(obstacle_rgb).double()

            save_image(torch.from_numpy(observations["rgb"]/255).float().permute(2,0,1), 'figures/rgb/img'+str(self.navigate_steps)+'.png')
            depth_normalized = (observations["depth"] - np.min(observations["depth"])) / (np.max(observations["depth"]) - np.min(observations["depth"]))
            depth_normalized = depth_normalized[None, :, :]
            save_image(torch.from_numpy(depth_normalized).float(), 'figures/depth/img'+str(self.navigate_steps)+'.png')
            # dist= torch.stack((torch.from_numpy(self.planner.fmm_dist), torch.from_numpy(self.planner.fmm_dist), torch.from_numpy(self.planner.fmm_dist)))
            # save_image((dist / dist.max()), 'figures/dist/img'+str(self.navigate_steps)+'.png')
            save_image((gray_map / gray_map.max()), 'figures/map/img'+str(self.navigate_steps)+'.png')
            save_image((torch.from_numpy(free_map) / free_map.max()), 'figures/free_map/img'+str(self.navigate_steps)+'.png')
            save_image(paper_map_trans.permute(2,0,1), 'figures/paper_map/img'+str(self.navigate_steps)+'.png')
            
            # exit()
        observations["pointgoal_with_gps_compass"] = self.get_relative_goal_gps(observations)

        ###-----------------------------------###

        self.last_loc = copy.deepcopy(self.full_pose)
        self.prev_action = number_action
        self.fake_prev_action = number_action
        if number_action == 0:
            a = 1
        self.navigate_steps += 1
        torch.cuda.empty_cache()
        
        return self.action_projection[number_action]
    
    
    def not_use_random_goal(self):
        self.move_since_random = 0
        self.using_random_goal = False
        
    def get_glip_real_label(self, prediction):
        labels = prediction.get_field("labels").tolist()
        new_labels = []
        if self.glip_demo.entities and self.glip_demo.plus:
            for i in labels:
                if i <= len(self.glip_demo.entities):
                    new_labels.append(self.glip_demo.entities[i - self.glip_demo.plus])
                else:
                    new_labels.append('object')
            # labels = [self.entities[i - self.plus] for i in labels ]
        else:
            new_labels = ['object' for i in labels]
        return new_labels       
    
    
    def fbe(self, traversible, start):
        # fontier: unknown area and free area 
        # unknown area: not free and not obstacle 
        # find nearest frontier and return the gps 
        fbe_map = torch.zeros_like(self.full_map[0,0])
        fbe_map[self.fbe_free_map[0,0]>0] = 1 # first free 
        fbe_map[skimage.morphology.binary_dilation(self.full_map[0,0].cpu().numpy(), skimage.morphology.disk(4))] = 3 # then dialte obstacle
    
        fbe_cp = copy.deepcopy(fbe_map)
        fbe_cpp = copy.deepcopy(fbe_map)
        fbe_cp[fbe_cp==0] = 4 # don't know space is 4
        fbe_cp[fbe_cp<4] = 0 # free and obstacle
        selem = skimage.morphology.disk(1)
        fbe_cpp[skimage.morphology.binary_dilation(fbe_cp.cpu().numpy(), selem)] = 0 # don't know space is 0 dialate unknown space
        
        diff = fbe_map - fbe_cpp
        frontier_map = diff == 1
        # get clostest frontier from the agent
        frontier_locations = torch.stack([torch.where(frontier_map)[0], torch.where(frontier_map)[1]]).T
        num_frontiers = len(torch.where(frontier_map)[0])
        if num_frontiers == 0:
            return None
        
        # for each frontier, calculate the inverse of distance
        planner = FMMPlanner(traversible, None)
        state = [start[0] + 1, start[1] + 1]
        planner.set_goal(state)
        fmm_dist = planner.fmm_dist[::-1]
        frontier_locations += 1
        frontier_locations = frontier_locations.cpu().numpy()
        distances = fmm_dist[frontier_locations[:,0],frontier_locations[:,1]] / 20
        idx_16 = np.where(distances>=1.2)
        distances_16 = distances[idx_16]
        distances_16_inverse = 1 - (np.clip(distances_16,0,11.2)-1.2) / (11.2-1.2)
        frontier_locations_16 = frontier_locations[idx_16]
        if len(distances_16) == 0:
            return None
        num_16_frontiers = len(idx_16[0])
        scores = np.zeros((num_16_frontiers))
                
        if self.args.reasoning in ['both', 'room']:
            if self.args.PSL_infer == 'optim':
                for predicate in self.psl_model.get_predicates().values():
                    if predicate.name() in ['ISNEARROOM', 'CHOOSE', 'SHORTDIST']:
                        predicate.clear_data()
            # for each frontier, calculate the rooms that within 0.6 meters of this frontier, at most 9 rooms
            for i, loc in enumerate(frontier_locations_16):
                sub_room_map = self.room_map[0,:,max(0,loc[0]-12):min(self.map_size-1,loc[0]+13), max(0,loc[1]-12):min(self.map_size-1,loc[1]+13)].cpu().numpy()
                whether_near_room = np.max(np.max(sub_room_map, 1),1) # 1*9
                if self.args.PSL_infer == 'optim':
                    # feed data
                    whether_near_room_list = whether_near_room.tolist()
                    data = pandas.DataFrame([[j, i, whether_near_room_list[j]] for j in range(len(whether_near_room_list))], columns = list(range(3)))
                    self.psl_model.get_predicate('IsNearRoom').add_data(Partition.OBSERVATIONS, data)
                else:
                    score_1 = np.clip(1-(1-self.prob_array_room)-(1-whether_near_room), 0, 10)
                    score_2 = 1- np.clip(self.prob_array_room+(1-whether_near_room), -10,1)
                    scores[i] = np.sum(score_1) - np.sum(score_2)
        
        #object near the frontier
        if self.args.reasoning in ['both', 'obj']:
            if self.args.PSL_infer == 'optim':
                for predicate in self.psl_model.get_predicates().values():
                    if predicate.name() in ['ISNEAROBJ', 'CHOOSE', 'SHORTDIST']:
                        predicate.clear_data()
            for i in range(len(categories_21)):
                num_obj = len(self.obj_locations[i])
                if num_obj <= 0:
                    continue
                frontier_location_mtx = np.tile(frontier_locations_16, (num_obj,1,1)) # k*m*2
                obj_location_mtx = np.array(self.obj_locations[i])[:,1:] # k*2

                # print(frontier_location_mtx.shape)
                # print(obj_location_mtx.shape)
                # exit()

                obj_confidence_mtx = np.tile(np.array(self.obj_locations[i])[:,0],(num_16_frontiers,1)).transpose(1,0) # k*m
                obj_location_mtx = np.tile(obj_location_mtx, (num_16_frontiers,1,1)).transpose(1,0,2) # k*m*2
                dist_frontier_obj = np.square(frontier_location_mtx - obj_location_mtx)
                dist_frontier_obj = np.sqrt(np.sum(dist_frontier_obj, axis=2)) / 20 # k*m
                near_frontier_obj = dist_frontier_obj < 1.2 # k*m 
                obj_confidence_mtx[near_frontier_obj==False] = 0 # k*m 
                obj_confidence_max = np.max(obj_confidence_mtx, axis=0)
                if self.args.PSL_infer == 'optim':
                    # feed data
                    whether_near_obj_list = obj_confidence_max.tolist()
                    data = pandas.DataFrame([[i, j, whether_near_obj_list[j]] for j in range(len(whether_near_obj_list)) if whether_near_obj_list[j] > 0], columns = list(range(3)))
                    self.psl_model.get_predicate('IsNearObj').add_data(Partition.OBSERVATIONS, data)
                else:
                    score_1 = np.clip(1-(1-self.prob_array_obj[i])-(1-obj_confidence_max), 0, 10)
                    score_2 = 1- np.clip(self.prob_array_obj[i]+(1-obj_confidence_max), -10,1)
                    scores += score_1 - score_2
                
        # select the frontier with highest score
        if self.args.PSL_infer != 'optim':
            if self.args.reasoning == 'both':
                scores += 2 * distances_16_inverse
            else:
                scores += 1 * distances_16_inverse
            idx_16_max = idx_16[0][np.argmax(scores)]
            goal = frontier_locations[idx_16_max] - 1
            # with open("output/FBE_PSL_oh_gpt_o/frontier_dist.txt", "a") as file_object:
            #     file_object.write(str(distances[idx_16_max]) + '\n')
        else:
            data = pandas.DataFrame([[i] for i in range(num_16_frontiers)], columns = list(range(1)))
            self.psl_model.get_predicate('Choose').add_data(Partition.TARGETS, data)
            
            data = pandas.DataFrame([[i, distances_16_inverse[i]] for i in range(num_16_frontiers)], columns = list(range(2)))
            self.psl_model.get_predicate('ShortDist').add_data(Partition.OBSERVATIONS, data)
            
            result = self.psl_model.infer(additional_cli_options = ADDITIONAL_CLI_OPTIONS, psl_config = ADDITIONAL_PSL_OPTIONS)
            for key, value in result.items():
                result_dt_frame = value
            
            scores = result_dt_frame.loc[:,'truth']
            idx_16_max = idx_16[0][np.argmax(scores)]
            goal = frontier_locations[idx_16_max] - 1
        return goal
        
    def get_goal_gps(self, observations, angle, distance):
        ### return goal gps in the original agent coordinates
        if type(angle) is torch.Tensor:
            angle = angle.cpu().numpy()
        agent_gps = observations['gps']
        agent_compass = observations['compass']
        goal_direction = agent_compass - angle/180*np.pi
        goal_gps = np.array([(agent_gps[0]+np.cos(goal_direction)*distance).item(),
         (agent_gps[1]-np.sin(goal_direction)*distance).item()])
        return goal_gps

    def get_relative_goal_gps(self, observations, goal_gps=None):
        if goal_gps is None:
            goal_gps = self.goal_gps
        direction_vector = goal_gps - np.array([observations['gps'][0].item(),observations['gps'][1].item()])
        rho = np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
        phi_world = np.arctan2(direction_vector[1], direction_vector[0])
        agent_compass = observations['compass']
        phi = phi_world - agent_compass
        return np.array([rho, phi.item()], dtype=np.float32)

    def glip_panorama(self, caption, thresh=None):
        """
        return a prediction on the panoramic view
        """
        assert len(self.panoramic) == 6
        predictions = self.glip_demo.inference_batch(self.panoramic, caption, thresh)

        for prediction in predictions:
            new_labels = self.get_glip_real_label(prediction)
            prediction.add_field("labels", new_labels)
        
        # combine same category of same position.    
        
        return predictions
   
    def init_map(self):
        self.map_size = self.map_size_cm // self.map_resolution
        full_w, full_h = self.map_size, self.map_size
        self.full_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.room_map = torch.zeros(1,9 ,full_w, full_h).float().to(self.device)
        self.visited = self.full_map[0,0].cpu().numpy()
        self.collision_map = self.full_map[0,0].cpu().numpy()
        self.fbe_free_map = copy.deepcopy(self.full_map).to(self.device) # 0 is unknown, 1 is free
        self.full_pose = torch.zeros(3).float().to(self.device)
        # Origin of local map
        self.origins = np.zeros((2))
        
        def init_map_and_pose():
            self.full_map.fill_(0.)
            self.full_pose.fill_(0.)
            # full_pose[:, 2] = 90
            self.full_pose[:2] = self.map_size_cm / 100.0 / 2.0  # put the agent in the middle of the map

        init_map_and_pose()

    def update_map(self, observations):
        """
        full pose: gps and angle in the initial coordinate system, where 0 is towards the x axis
        """
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+torch.from_numpy(observations['gps']).to(self.device)[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-torch.from_numpy(observations['gps']).to(self.device)[1]
        self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self.full_map = self.sem_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.full_map)
    
    def update_free_map(self, observations):
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+torch.from_numpy(observations['gps']).to(self.device)[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-torch.from_numpy(observations['gps']).to(self.device)[1]
        self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self.fbe_free_map = self.free_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.fbe_free_map)
        self.fbe_free_map[0,0,int(self.map_size_cm / 10) - 10:int(self.map_size_cm / 10) + 11, int(self.map_size_cm / 10) - 10:int(self.map_size_cm / 10) + 11] = 1
    
    def update_room_map(self, observations, room_prediction_result):
        """
        new: refine the room map by set living room to 0 if there is other rooms
        """
        new_room_labels = self.get_glip_real_label(room_prediction_result)
        print("new room labels: ", new_room_labels) #['bedroom', 'room']
        type_mask = np.zeros((9,self.depth.shape[0], self.depth.shape[1]))
        bboxs = room_prediction_result.bbox
        score_vec = torch.zeros((9)).to(self.device)
        for i, box in enumerate(bboxs):
            box = box.to(torch.int64)
            try:
                idx = rooms.index(new_room_labels[i])
            except:
                continue
            type_mask[idx,box[1]:box[3],box[0]:box[2]] = 1
            score_vec[idx] = room_prediction_result.get_field("scores")[i]
        '''update room location'''
        self.room_map = self.room_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.room_map, torch.from_numpy(type_mask).to(self.device).type(torch.float32), score_vec)
        self.room_map_refine = copy.deepcopy(self.room_map)
        other_room_map_sum = self.room_map_refine[0,0] + torch.sum(self.room_map_refine[0,2:],axis=0)
        self.room_map_refine[0,1][other_room_map_sum>0] = 0
    
    def get_traversible(self, map_pred, pose_pred):
        grid = np.rint(map_pred)

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        r, c = start_y, start_x
        start = [int(r*100/self.map_resolution - gy1),
                 int(c*100/self.map_resolution - gx1)]
        # start = [int(start_x), int(start_y)]
        start = pu.threshold_poses(start, grid.shape)
        self.visited[gy1:gy2, gx1:gx2][start[0]-2:start[0]+3,
                                       start[1]-2:start[1]+3] = 1
        #Get traversible
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        def delete_boundary(mat):
            new_mat = copy.deepcopy(mat)
            return new_mat[1:-1,1:-1]
        
        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        selem = skimage.morphology.square(4)
        traversible = skimage.morphology.binary_dilation(
                    grid[y1:y2, x1:x2],
                    selem) != True

        if not(traversible[start[0], start[1]]):
            print("Not traversible, step is  ", self.navigate_steps)

        # obstacle dilation do not dilate collision
        traversible = 1 - traversible
        selem = skimage.morphology.disk(1)
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem) != True
        
        traversible[int(start[0]-y1)-1:int(start[0]-y1)+2,
            int(start[1]-x1)-1:int(start[1]-x1)+2] = 1
        traversible = traversible * 1.
        
        traversible[self.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 1
        traversible[self.collision_map[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 0 # temp change to see
        traversible = add_boundary(traversible)
        return traversible, start, start_o
    
    def _plan(self, traversible, goal_map, agent_pose, start, start_o, goal_found):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        # if newly_goal_set:
        #     self.action_5_count = 0

        if self.prev_action == 1:
            x1, y1, t1 = self.last_loc.cpu().numpy()
            x2, y2, t2 = self.full_pose.cpu()
            y1 = self.map_size_cm/100 - y1
            y2 = self.map_size_cm/100 - y2
            t1 = -t1
            t2 = -t2
            buf = 4
            length = 5

            if abs(x1 - x2)< 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 1
                self.col_width = min(self.col_width, 3)
            else:
                self.col_width = 1
            # self.col_width = 4
            dist = pu.get_l2_distance(x1, x2, y1, y2)
            col_threshold = self.collision_threshold

            if dist < col_threshold: # Collision
                self.former_collide += 1
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05*((i+buf) * np.cos(np.deg2rad(t1)) + \
                                        (j-width//2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05*((i+buf) * np.sin(np.deg2rad(t1)) - \
                                        (j-width//2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(round(r*100/self.map_resolution)), \
                               int(round(c*100/self.map_resolution))
                        [r, c] = pu.threshold_poses([r, c],
                                    self.collision_map.shape)
                        self.collision_map[r,c] = 1
            else:
                self.former_collide = 0

        stg, stop, = self._get_stg(traversible, start, np.copy(goal_map), goal_found)

        # Deterministic Local Policy
        if stop:
            action = 0
            (stg_y, stg_x) = stg

        else:
            (stg_y, stg_x) = stg
            angle_st_goal = math.degrees(math.atan2(stg_y - start[0],
                                                stg_x - start[1]))
            angle_agent = (start_o)%360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_st_goal- angle_agent)%360.0
            if relative_angle > 180:
                relative_angle -= 360
            if self.former_collide < 10:
                if relative_angle > 16:
                    action = 3 # Right
                elif relative_angle < -16:
                    action = 2 # Left
                else:
                    action = 1
            elif self.prev_action == 1:
                if relative_angle > 0:
                    action = 3 # Right
                else:
                    action = 2 # Left
            else:
                action = 1
            if self.former_collide >= 10 and self.prev_action != 1:
                self.former_collide  = 0
            if stg_y == start[0] and stg_x == start[1]:
                action = 1

        return stg_y, stg_x, action
    
    def _get_stg(self, traversible, start, goal, goal_found):
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        def delete_boundary(mat):
            new_mat = copy.deepcopy(mat)
            return new_mat[1:-1,1:-1]
        
        goal = add_boundary(goal, value=0)
        original_goal = copy.deepcopy(goal)
        
            
        centers = []
        if len(np.where(goal !=0)[0]) > 1:
            goal, centers = CH._get_center_goal(goal)
        state = [start[0] + 1, start[1] + 1]
        self.planner = FMMPlanner(traversible, None)
            
        if self.dilation_deg!=0: 
            #if self.args.debug_local:
            #    self.print_log("dilation added")
            goal = CH._add_cross_dilation(goal, self.dilation_deg, 3)
            
        if goal_found:
            # if self.args.debug_local:
            #     self.print_log("goal found!")
            try:
                goal = CH._block_goal(centers, goal, original_goal, goal_found)
            except:
                goal = self.set_random_goal(goal)

        self.planner.set_multi_goal(goal, state) # time cosuming 

        decrease_stop_cond =0
        if self.dilation_deg >= 6:
            decrease_stop_cond = 0.2 #decrease to 0.2 (7 grids until closest goal)
        stg_y, stg_x, replan, stop = self.planner.get_short_term_goal(state, found_goal = goal_found, decrease_stop_cond=decrease_stop_cond)
        stg_x, stg_y = stg_x - 1, stg_y - 1
        if stop:
            a = 1
        
        # self.closest_goal = CH._get_closest_goal(start, goal)
        
        return (stg_y, stg_x), stop
    
    def set_random_goal(self):
        obstacle_map = self.full_map.cpu().numpy()[0,0,::-1]
        goal = np.zeros_like(obstacle_map)
        goal_index = np.where((obstacle_map<1))
        np.random.seed(self.total_steps)
        if len(goal_index[0]) != 0:
            i = np.random.choice(len(goal_index[0]), 1)[0]
            h_goal = goal_index[0][i]
            w_goal = goal_index[1][i]
        else:
            h_goal = np.random.choice(goal.shape[0], 1)[0]
            w_goal = np.random.choice(goal.shape[1], 1)[0]
        goal[h_goal, w_goal] = 1
        return goal


def build():
    agent_class = CLIP_LLM_FMMAgent_NonPano
    agent_kwargs = {}
    # resembles SimpleRandomAgent(**{})
    render_depth = True
    return agent_class, agent_kwargs, render_depth
