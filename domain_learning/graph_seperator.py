import concurrent.futures
from typing import List
#import pymimir 
import argparse
import os
import networkx as nx
import itertools
import clingo
import copy
import time
import multiprocessing as mp
import concurrent
import random
from collections import defaultdict
from pathlib import Path
#from pymimir import PDDLParser, StateSpace, PDDLParser
from clingo.control import Control
from itertools import chain, combinations
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed, wait
from graph_generator import get_trace_rl, get_trace_simple, get_nx_graph_only_action_names_new
from graph_generator import bfs_state_space, get_nx_graph_from_state_space
from py_separator_utils.mimir_holder import mimir_holder
import sys


def get_arguments(): 
    parser = argparse.ArgumentParser('graph_generator.py')  
    parser.add_argument("-d", "--domain", type=Path, required=True, help="specify domain that is in the pddl_files folder.")
    parser.add_argument("-i", "--instance", type=Path, nargs='+', required=True, help="specify list of instances that is in the pddl_files folder.")
    parser.add_argument("-v", "--verification_instance", type=str, action='append', required=False, help="specify list of instances that is in the pddl_files folder.")
    parser.add_argument("-p", "--processes", type=int, required=True, help="number of max. parallel processes, 1 means sequential algorihtm")
    parser.add_argument("-o", "--output", type=str, required=False, help='name of output file')
    parser.add_argument("-lm", "--learning_mode", type=str, required=False, default='fg', 
                        help='Defines the input to the learinig alg. \n fg = full graphs (default)\n pg = partial graphs\n st= simple traces\n rl= rl style traces')
    # parser.add_argument("-vm", "--verification_mode", type=str, required=False, default='fg', 
    #                     help='Defines the input to the learinig alg. \n fg = full graphs (default)\n pg = partial graphs\n st= simple traces\n rl= rl style traces')
    parser.add_argument("-ls", "--learning_size", type=int, required=False, help="size of the input if mode is not fg")
    # parser.add_argument("-vs", "--verification_size", type=int, required=False, help="size of the input if mode is not fg")
    parser.add_argument("-ln", "--learning_number_inputs", type=int, required=False, default=1, help="number of sampled inputs if mode is not fg")
    # parser.add_argument("-vn", "--verification_number_inputs", type=int, required=False, default=1, help="number of sampled inputs if mode is not fg")
    # parser.add_argument("-vt", "--verification_termination", action=argparse.BooleanOptionalAction, required=False, help="If set the verification stops at the first wrong predicate.")

    # parse arguments
    args = parser.parse_args()
    return args


def get_verification_instances(verification_input):
    
    instances = []
    modes = ['fg', 'st', 'rl', 'pg', 'nfg', 'nst', 'nrl', 'npg']

    for instance in verification_input:

        split_input = instance.split(',')

        if 1 >= len(split_input) or len(split_input) > 5:
            print(len(split_input))
            print('Length of input {} does not fit!'.format(instance))
            continue

        if not os.path.exists(split_input[0]):
            print('For input {} the path {} does not exist'.format(instance, split_input[0]))
            continue

        if not split_input[1] in modes:
            print('For input {} mode {} does not exist!'.format(instance, split_input[1]))
            continue

        if split_input[1] in ['st', 'rl', 'nst', 'nrl', 'pg', 'npg'] and len(split_input) < 3:
            print('For input {} no specification of input size!'.format(instance))
            continue

        if len(split_input) >= 3:
            split_input[2] = int(split_input[2])
            if split_input[2] < 1:
                print('No valid number of nodes!')
                continue

        if len(split_input) >= 4:
            split_input[3] = int(split_input[3])
            if split_input[3] < 1:
                print('No valid number of traces!')
                continue

        if len(split_input) == 5:
            split_input_val_5 = int(split_input[4])
            if split_input_val_5 == 0:
                split_input[3] = False
            elif split_input_val_5 == 1:
                split_input[3] = True
            else:
                print('No valid truth value for early termination!')
                continue

        vObject = VerificationObject(*split_input)

        instances.append(vObject)

    return instances


class VerificationObject:

    def __init__(self, instance, mode, input_size=0, number_traces=1, early_termination=True) -> None:
        self.instance = instance
        self.mode = mode
        self.number_traces = number_traces
        self.input_size = input_size
        self.early_termination = early_termination

    def get_instance(self):
        return self.instance

    def get_mode(self):
        return self.mode

    def get_number_traces(self):
        return self.number_traces

    def get_input_size(self):
        return self.input_size

    def get_early_termination(self):
        return self.early_termination
        


def powerset_without_empty_set(iterable):
    "powerset([1,2,3]) → (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class ColorSplit:

    def __init__(self, possible_pattern):
        #possible_pattern is a subset of the powerset of all patterns per tc
        #possible pattern contains the formally selected patterns of an feature
        self.splits = []
        self.contained_pattern = set()
        self.possible = True
        self.possible_pattern = possible_pattern

    def get_split(self):
        if self.possible:
            return self.splits
        return None
    
    def get_possible_splits(self):

        if not self.possible:
            return []

        not_seen_seps = copy.deepcopy(self.possible_pattern)
        for _split in self.splits:
            for _sep in _split[0]:
                not_seen_seps.discard(_sep)
            for _sep in _split[1]:
                not_seen_seps.discard(_sep)     
            
        if len(self.splits) == 1 and len(not_seen_seps) == 0:
            return self.splits
        elif len(self.splits) == 0 and len(not_seen_seps) == 0:
            return []
        else:
            # 1. get all possible splits based on seen separators
            all_splits = []
            if len(self.splits) == 1:
                all_splits = copy.deepcopy(self.splits)
            elif len(self.splits) == 0:
                all_splits = [[set(),set()]]
            else:
                # https://stackoverflow.com/questions/44730234/generate-all-possible-splits-of-a-list-in-python
                # goes back to binary counting 
                subsets = [contained for l in range(len(self.splits)) for contained in itertools.combinations(range(len(self.splits)), r=l)]
                for i in range(len(subsets)//2+1):
                    combi = subsets[i]

                    split0, split1 = set(), set()
                    for split_num, split in enumerate(self.splits):
                        if split_num in combi:
                            split0 = split[0] | split0
                            split1 = split[1] | split1
                        else:
                            split0 = split[1] | split0
                            split1 = split[0] | split1
                    all_splits.append([split0, split1])
                    
            
            # 2. get splits based on separators not seen
            new_list = [] 
            if len(not_seen_seps) == 0:
                return all_splits
            else:
                # we need each subset on each side since the splits in all splits will make the union unambigous 
                if len(self.splits) > 0: 
                    for not_seen_0 in powerset(not_seen_seps):
                        not_seen_set_0 = set(not_seen_0)
                        not_seen_set_1 = not_seen_seps - not_seen_set_0
                        #if len(not_seen_0) + len(not_seen_set_1) > 0:
                        #    not_seen_set_1 = set(not_seen_1)
                        for split in all_splits:
                            
                            set_0 = not_seen_set_0 | split[0]
                            set_1 = not_seen_set_1 | split[1]
                            new_list.append([set_0, set_1])
                else:
                    # if there is no split we introduce redundancy if we use all possibile subsets on each side
                    # TODO this can be done more efficient, only a quick and dirty fix 
                    all_seen_sets = set()
                    for not_seen_0 in powerset(not_seen_seps):
                        not_seen_set_0 = set(not_seen_0)
                        not_seen_set_1 = not_seen_seps - not_seen_set_0

                        if not frozenset(not_seen_set_1) in all_seen_sets:
                            if not frozenset(not_seen_set_0) in all_seen_sets:
                                new_list.append([not_seen_set_0, not_seen_set_1])
                                all_seen_sets.add(frozenset(not_seen_set_0))
                                all_seen_sets.add(frozenset(not_seen_set_1))

                return new_list

    def impossible(self):
        self.splits = None
        self.contained_pattern = None
        self.possible = False

    def add(self, color_set_1: set, color_set_2: set):
        if not self.possible:
            return False
        x = len(color_set_1) + len(color_set_2)
        if x == 0:
            return True
        else:
            contained_1 = len(self.contained_pattern.intersection(color_set_1))
            contained_2 = len(self.contained_pattern.intersection(color_set_2))

            if 0 == contained_1 + contained_2:
                self.splits.append([color_set_1, color_set_2])
            else:
                if len(self.splits) == 1:
                    if len(color_set_1) > 0:
                        if self.splits[0][0].isdisjoint(color_set_1) and self.splits[0][1].isdisjoint(color_set_2):
                            self.splits[0][1].update(color_set_1)
                            self.splits[0][0].update(color_set_2)
                        elif self.splits[0][0].isdisjoint(color_set_2) and self.splits[0][1].isdisjoint(color_set_1):
                            self.splits[0][0].update(color_set_1)
                            self.splits[0][1].update(color_set_2)
                        else:
                            self.possible = False
                            return False
                    else:
                        if self.splits[0][0].isdisjoint(color_set_2):
                            self.splits[0][0].update(color_set_1)
                            self.splits[0][1].update(color_set_2)
                        elif self.splits[0][1].isdisjoint(color_set_2):
                            self.splits[0][1].update(color_set_1)
                            self.splits[0][0].update(color_set_2)
                        else:
                            self.possible = False
                            return False
                else:
                    delete_index = []
                    for split_num, split in enumerate(self.splits):
                        if not split[0].isdisjoint(color_set_1) or not split[1].isdisjoint(color_set_2):
                            color_set_1.update(split[0])
                            color_set_2.update(split[1])
                            delete_index.append(split_num)
                        elif not split[1].isdisjoint(color_set_1) or not split[0].isdisjoint(color_set_2):
                            color_set_1.update(split[1])
                            color_set_2.update(split[0])
                            delete_index.append(split_num)
                        if not color_set_1.isdisjoint(color_set_2):
                            self.possible = False
                            break
                    if self.possible:
                        delete_index.reverse()
                        for index in delete_index:
                            self.splits.pop(index)
                        self.splits.append([color_set_1, color_set_2])
            if self.possible:
                for _pattern in color_set_1:
                    self.contained_pattern.add(_pattern)
                for _pattern in color_set_2:
                    self.contained_pattern.add(_pattern)
                return True
            else:
                self.impossible()
                return False

class BenchmarkValue:

    def __init__(self, pattern_combis) -> None:
        self.graphs = 0
        self.pattern = 0
        if pattern_combis is None:
            self.pattern_combinations = set()
        else: 
            self.pattern_combinations = pattern_combis
        self.passed_pattern_combination = 0
        self.viewed_candidates = 0
        self.type_combinations = 0
        self.max_patter_per_combination = 0
        self.final_feature_nums = 0

    def __str__(self):
        output = '#graphs = {},  #type_combinations {}\n'.format(self.graphs, self.type_combinations)
        output += '#pattern = {}, #pattern_combi = {}, max_pattern_per_combi = {}, #passed_combi = {}, #viewed_candidates = {}'.format(
            self.pattern, len(self.pattern_combinations), self.max_patter_per_combination, self.passed_pattern_combination, self.viewed_candidates
        )
        return output

    def add_graph(self):
        self.graphs += 1
    
    def add_pattern(self):
        self.pattern += 1

    def add_pattern_combi(self, pattern_combi):
        self.pattern_combinations.add(pattern_combi)

    def add_passed_pattern(self):
        self.passed_pattern_combination += 1

    def sum_passed_pattern(self, addition):
        self.passed_pattern_combination += addition

    def add_viewed_candidates(self):
        self.viewed_candidates += 1

    def add_type_combinations(self):
        self.type_combinations += 1

    def add_max_pattern_per_combination(self, number: int):
        self.max_patter_per_combination = max(self.max_patter_per_combination, number)

    def add_final_feature_num(self):
        self.final_feature_nums += 1

    def get_graph(self):
        return self.graphs
    
    def get_pattern(self):
        return self.pattern

    def get_pattern_combi(self):
        return self.pattern_combinations
    
    def get_pattern_combi_length(self):
        return len(self.pattern_combinations)

    def get_passed_pattern(self):
        return self.passed_pattern_combination

    def get_viewed_candidates(self):
        return self.viewed_candidates
    
    def get_type_combinations(self):
        return self.type_combinations
    
    def get_max_pattern_per_combination(self, number: int):
        return self.max_patter_per_combination

    def get_final_feature_num(self):
        return self.final_feature_nums

'''
# TODO currently not used
# Function that genereated subproblem specific clingo input 
def create_clingo_input(pddl_problem, pddl_parser):

    # get list of all states
    states_objects = pddl_problem.get_states()
    states = [pddl_problem.get_state_index(state) for state in states_objects]

    # get set of all actions 
    action_list = {str(trans.get_creating_action()) for state in states for trans in pddl_problem.get_forward_transitions(state)}

    # set for objects, dict for actions
    objects, action_arity = set(), dict()

    # get all objects in the instance 
    for obj in pddl_parser.get_problem().get_objects(): 
        objects.add(obj.get_name())

    # get all actions in the domain 
    for act in pddl_parser.get_domain().get_actions(): 
        action_arity[act.get_name()] = act.get_arity()

    # map actions to integers 
    action_mapping = {action:str(i+1) for i,action in enumerate(sorted(list(action_arity.keys())))}
    object_mapping = {obj:str(i+1) for i,obj in enumerate(sorted(list(objects)))}
    object_mapping[-1] = '-1'

    # non_applicable_actions contains all actions that are not in the state space, negative_actions contains the actions in clingo format
    non_applicable_actions, negative_actions = set(), []

    # get all not possible actions
    for act, arity in action_arity.items():
        # check for each possible combination of objects if there is an action with this objects 
        #for pattern_set in itertools.permutations(range(arity), arity): 
        for obj_set in itertools.product(objects,repeat=arity):
    
            grounded_action = '(' + act + ' ' + ' '.join(obj_set) + ')'
            # check wheteher action is possible 
            if grounded_action not in action_list:
                # append -1 to the list such that there all actions have object tuples of length 3 
                ob = list(obj_set)
                while len(ob) < 3: 
                    ob.append(-1)

                # create action for clingo input 
                neg_action = 'neg_action(' + action_mapping[act] + ',(' + ','.join([object_mapping[o] for o in ob]) + ')).\n'

                # probably unneccessary since we check each combination only once 
                if grounded_action not in non_applicable_actions:
                    # add action to the output and to the non applicable set 
                    non_applicable_actions.add(grounded_action)
                    negative_actions.append(neg_action)


    output = ''
    max_arity = max(list(action_arity.values()))


    # add each possible object tuple to the clingo input 
    for i in range(1,max_arity+1):
        arity_str = 'obj_tuple(' + str(i) + ','
        for obj in itertools.product(objects,repeat=i): 
            output += arity_str + '(' + ','.join([object_mapping[o] for o in obj]) + ')).\n'

    # add each possible pattern for each action to the input 
    # -- pattern(index,action,arity,pattern)
    pattern_dict = {}
    pattern_index = 1
    for action,arity in action_arity.items():
        for ar in range(1,arity+1):
            for tup in itertools.product(set(range(arity)),repeat=ar):
                output += 'pattern(' + str(pattern_index) + ',' + action_mapping[action] + ',' + str(ar) + ',(' + ','.join([str(t) for t in tup]) + ')).\n'
                pattern_dict[pattern_index] = [ar, action, tup]
                pattern_index += 1 

    # add all possible actions to the clingo input 
    # add -1 to the end of all actions which have less then three arguments (maybe better with max_arity instead of 3)
    for state in states:
        for trans in pddl_problem.get_forward_transitions(state): 
            action = trans.get_creating_action()
            obj = list([object_mapping[obj.get_name()] for obj in action.get_objects()])
            while len(obj) < 3: 
                obj.append(str(-1))

            output += 'pos_action(' + action_mapping[action.get_name()] + ',(' + ','.join(obj) + ')).\n'

    # add all negative actions to the clingo input 
    for neg_action in negative_actions: 
        output += neg_action

    # add all objects to the clingo input 
    for obj in objects:
        output += 'obj(' + object_mapping[obj] + ').\n'

    # TODO location where clingo input is stored 
    out_path = str(os.path.join(os.path.dirname(os.path.realpath(__file__)))) + '/lp_input/' + pddl_parser.get_domain().get_name() + '.lp'

    # write output to file 
    with open(out_path , "w") as text_file:
        text_file.write(output)


    # convert action mapping and object mapping to int values
    for action_name, mapping_value in action_mapping.items():
        action_mapping[action_name] = int(mapping_value)

    for object_name, mapping_value in object_mapping.items():
        object_mapping[object_name] = int(mapping_value)

    return action_mapping, object_mapping, out_path, pattern_dict
'''

'''
def get_statics_with_pattern(optimal_solution):

    statics = dict()  # dictionary that stores: key = static, value = ariry  
    statics_pattern = dict()  # dictionary that stores: key = static, values = all patterns of this value
    
    if optimal_solution is not None:
        print('optimal_solution',optimal_solution, optimal_solution.type)

        # add all statics to the dict and create empty set for the pattern it contains 
        for opt in optimal_solution.symbols(shown=True):
            if str(opt.name) == 'static':
                statics[opt.arguments[0].number] = opt.arguments[1].number
                statics_pattern[opt.arguments[0].number] = set()

        # add all contained patterns for all statics     
        for opt in optimal_solution.symbols(shown=True):
            if str(opt.name) == 'pattern':
                statics_pattern[opt.arguments[0].number].add(opt.arguments[1].number)

        return statics, statics_pattern

    return None, None
'''

def learn_locm_type(all_grounded_actions, action_arity, object_mapping, previous_action_types):

    object_positions = {_obj: set() for _obj in object_mapping.values()}

    for _act in all_grounded_actions:
        for _pos, _obj in enumerate(_act[1]):
            object_positions[_obj].add((_act[0], _pos))

    print('OBJECT POSTIIONS', object_positions)

    if previous_action_types is None:
        action_types = {(_act, _pos):None for _act, _arity in action_arity.items() for _pos in range(_arity)}
        for _num, _act_type in enumerate(action_types):
            action_types[_act_type] = _num
    else:
        action_types = previous_action_types

    for object, action_positions in object_positions.items():
        #all_types = {action_types[_act_type] for _act_type in action_positions}
        if len(action_positions) > 1:
            all_index = {action_types[x] for x in action_positions}
            new_index = min(all_index)
            for _act_type in action_types.keys():
                if action_types[_act_type] in all_index:
                    action_types[_act_type] = new_index

    return action_types



def get_locm_types(all_grounded_actions, object_mapping, locm_types):

    all_objects = {_obj for _obj in object_mapping.values()}
    object_type = {_obj: None for _obj in object_mapping.values()}

    for _act in all_grounded_actions:
        for _pos, _obj in enumerate(_act[1]):
            object_type[_obj] = locm_types[(_act[0], _pos)]
            all_objects.discard(_obj)
        if len(all_objects) == 0:
            break

    return object_type

def verify_locm_types(all_grounded_actions, object_mapping, locm_types):

    object_type = {_obj: set() for _obj in object_mapping.values()}

    for _act in all_grounded_actions:
        for _pos, _obj in enumerate(_act[1]):
            object_type[_obj].add(locm_types[(_act[0], _pos)])

    for _obj, _obj_types in list(object_type.items()):
        if len(_obj_types) > 1:
            return None
        elif len(_obj_types) == 0:
            del object_type[_obj]
            pos = find_key(object_mapping, _obj)
            del object_mapping[pos]
        else:
            object_type[_obj] = list(_obj_types)[0]

    return object_type, object_mapping


def count_occurences(action_set, object_mapping, arity, action_mapping):

    # create dictionaty that contains as keys types and as values the corresponding objects
    # output_dict = {_act: defaultdict(lambda: 0) for _val in sep.values() for _act in _val}
    output_dict = dict()

    # iterate over all possible actions in this domain
    for action, occurences in action_set.items():

        # get list of action objects
        cur_objects = list(action[1])

        # get all combinations (of length arity) of this objects (w.r.t. position of the objects)
        for tup in itertools.permutations(range(len(cur_objects)), r=arity):

            # get (action, pattern) tuple  
            _cur_action = (action[0], tuple(tup))

            # get object tuple 
            _cur_obj_tuple = tuple([cur_objects[i] for i in tup])

            # if (action, pattern) increase number of occurences of these objects
            if _cur_action in output_dict.keys():
                output_dict[_cur_action][_cur_obj_tuple] += occurences
            else:
                output_dict[_cur_action] = defaultdict(lambda:0)
                output_dict[_cur_action][_cur_obj_tuple] += occurences
    
    #print('arity', output_dict.keys())

    return output_dict


# Function that copys a graph
# Needed since some property is missing in pymimir that makes graph with pymimir lables not possible to copy
def copy_graph(graph):

    copy_G = nx.DiGraph()
    for n in list(graph.nodes()):
        copy_G.add_node(n)
    for edge in graph.edges(data='action'):
        data = copy.deepcopy(edge[2])
        copy_G.add_edge(edge[0], edge[1], action=data)

    return copy_G

def find_key(input_dict, value):
    for key, val in input_dict.items():
        if val == value:
            return key
    return "None"


# fuction that gets all possible seperators that belong to actions that formed self-loops
def get_false_seperator(seperators, delted_possible_actions, max_grounding, false_sep_edges, pattern_groundings):

    false_seps = set()

    for _sep in seperators:
        for _act in delted_possible_actions:
            if max_grounding in pattern_groundings[_sep]:
                if _act[2] in pattern_groundings[_sep][max_grounding]:
                    false_seps.add(_sep)
                    if _sep in false_sep_edges:
                        false_sep_edges[_sep].add((_act[0], _act[1]))
                    else:
                        false_sep_edges[_sep] = {(_act[0], _act[1])}
        

    return false_seps, false_sep_edges


# function that merges nodes in the graph based on lists of (u,v,action)
# also deletes ALL self-loops, 
# for self-loops with actions that shouldn't be removed are added to the output
# TODO here is probabaly performance imrovement possible
def merge_graph(Gx, node_mapping, actions_to_merge, actions_to_stay):

    # list of edges deleted that were not in actions_to_merge
    self_loops_not_possible = set()

    #equivalent_actions = set()

    # iterate over all edges were the nodes should be merged
    for (u,v) in actions_to_merge:
        
        if u not in node_mapping:
            print(node_mapping)

        map_u = node_mapping[u]
        while map_u != node_mapping[map_u]:
            map_u = node_mapping[map_u]

        if v not in node_mapping:
            print(node_mapping)

        map_v = node_mapping[v]
        while map_v != node_mapping[map_v]:
            map_v = node_mapping[map_v] 

        # only merge if not already merged
        if map_u != map_v:

            #for e in list(Gx.out_edges([map_u], keys=True, data='action')):
            #    if e[0] == map_u and e[1] == map_v and e[3] in actions_to_stay:
            #        self_loops_not_possible.add((map_u,map_u,e[3]))
            #for e in list(Gx.in_edges([map_u], keys=True, data='action')):
            #    if e[0] == map_u and e[1] == map_v and e[3] in actions_to_stay:
            #        self_loops_not_possible.add((map_u,map_u,e[3]))
            for edge in list(Gx.out_edges([map_u], data='action')):
                if Gx.has_edge(map_v,edge[1]):
                    #for _act1 in Gx.edges[map_u, edge[1]]['action']:
                        #for _act2 in Gx.edges[map_v, edge[1]]['action']:
                            #equivalent_actions.add((_act1, _act2))
                    Gx.edges[map_u, edge[1]]['action'].update(Gx.edges[map_v, edge[1]]['action'])

                if edge[1] == map_v:
                    for _act in edge[2]:
                        if _act in actions_to_stay:
                            self_loops_not_possible.add((map_u, map_v, _act))

            for edge in list(Gx.in_edges([map_u], data='action')):
                if Gx.has_edge(edge[0],map_v):
                    #for _act1 in Gx.edges[edge[0], map_u]['action']:
                        #for _act2 in Gx.edges[edge[0], map_v]['action']:
                            #equivalent_actions.add((_act1, _act2))
                    Gx.edges[edge[0], map_u]['action'].update(Gx.edges[edge[0], map_v]['action'])
                if edge[0] == map_v:
                    for _act in edge[2]:
                        if _act in actions_to_stay:
                            self_loops_not_possible.add((map_u, map_v, _act))

            for _x in list(Gx.edges[(map_u, map_v)]['action']):
                if _x in actions_to_stay:
                    self_loops_not_possible.add((map_u, map_u, _x))

            # update node mapping
            node_mapping[map_v] = map_u

            # merge the nodes adjacent to the edge
            # for a directed edge (u,v) the node v will be merged into the node u 
            # Gx = nx.contracted_edge(Gx, edge, self_loops=True, copy=False)
            nx.contracted_nodes(Gx, map_u, map_v, False, False)
            
    # return merged graph, updated mapping, all deleted actions that should not be deleted 
    return Gx, node_mapping, self_loops_not_possible, set()

        

def merge_graph_with_grounding(cur_grounding, G, node_mapping, edges_to_delete, actions_possible, already_del_actions, possible_action_pattern, pattern_groundings, possible_pattern):

    G, node_mapping, delted_possible_actions, equiv_actions = merge_graph(G, node_mapping, edges_to_delete, actions_possible)

    if already_del_actions is not None:
        for _del_action in already_del_actions:
            if _del_action[2] in actions_possible:
                delted_possible_actions.add(_del_action)

    G_copy = copy_graph(G)
    node_mapping_copy = copy.deepcopy(node_mapping)
    delted_possible_actions_copy = copy.deepcopy(delted_possible_actions)

    output_for_merged_graph_dict = [G_copy, node_mapping_copy, delted_possible_actions_copy]

    seperators = set(possible_pattern)
    #new_separators = set(possible_pattern)
    dead_separators = set()

    false_sep_with_edges = dict()

    # while len(G.nodes()) > 1 and len(delted_possible_actions) > 0:
    while len(delted_possible_actions) > 0:
        false_seps, false_sep_with_edges = get_false_seperator(seperators, delted_possible_actions, cur_grounding, false_sep_with_edges, pattern_groundings)

        # remove them from the seperators set 
        seperators = seperators - false_seps

        for false_sep in false_seps:
            dead_separators.add(false_sep)

        if len(seperators) == 0:
            break

        # get actions that need to stay
        actions_possible_for_grounding = set()
        for _action_pattern in seperators:
            if cur_grounding in pattern_groundings[_action_pattern]:
                for _possible_act in pattern_groundings[_action_pattern][cur_grounding]:
                    actions_possible_for_grounding.add(_possible_act)

        actions_not_possible_for_grounding = set()
        for _action_pattern in false_seps:
            if cur_grounding in pattern_groundings[_action_pattern]:
                for _possible_act in pattern_groundings[_action_pattern][cur_grounding]:
                    actions_not_possible_for_grounding.add(_possible_act)

        edges_to_merge, actions_to_stay = set(), set()
        for edge in G.edges(data='action'):
            #if len(edge[2].intersection(actions_possible_for_grounding)) != len(edge[2]):
            #    edges_to_merge.add((edge[0], edge[1]))
            if len(edge[2].intersection(actions_not_possible_for_grounding)) > 0:
                edges_to_merge.add((edge[0], edge[1]))

        # if there is no edge to remove we can exit the loop
        # if there is no separator left we can leave the loop
        if len(edges_to_merge) == 0:
            break
        
        # merge the nodes of the separators that formed self-loops 
        G, node_mapping, delted_possible_actions, new_equiv = merge_graph(G,node_mapping, edges_to_merge, actions_possible_for_grounding)

        equiv_actions.update(new_equiv)

    if len(seperators) == 0 or len(G.nodes) == 1:
        type_combination_possible = False
        return None, seperators, dead_separators, output_for_merged_graph_dict

    graph_list = list(G.edges(data='action'))

    return [graph_list, node_mapping, delted_possible_actions, false_sep_with_edges], seperators, dead_separators, output_for_merged_graph_dict
    return [G, node_mapping, delted_possible_actions, false_sep_with_edges], seperators, output_for_merged_graph_dict
    merged_graphs_for_grounding[cur_grounding] = [G, node_mapping, delted_possible_actions, false_sep_with_edges]
    all_seperators.append(seperators)


def get_graph_from_list(graph_list):

    G = nx.DiGraph()
    for edge in graph_list:
        G.add_edge(edge[0], edge[1], action=edge[2])

    return G

def check_separator(_cur_sep, merged_graphs_for_grounding, action_mapping, object_mapping, possible_separators, pattern_groundings, print_v, split):

    #_cur_sep is a subset of the powerset of all patterns of this TC
    #_cur_sep represents the formally selected patterns of a feature
    #split is a ColorSplit object containing information about the previous colorings for the feature
    #merged_graphs_for_grounding is a dict containing all graphs premerged for the key grounding

    _cur_sep_possible = True
    _cur_sep_colors = dict()

    for grounding, [G, node_mapping, _, _] in merged_graphs_for_grounding.items():

        coloring = get_coloring(G, grounding, _cur_sep, action_mapping, object_mapping, possible_separators[grounding], pattern_groundings)

        if coloring is None:
            _cur_sep_possible = False
            break

        _cur_sep_colors[grounding] = coloring

    #print(_cur_sep, _cur_sep_possible)

    if _cur_sep_possible:
        
        #split = ColorSplit(set(_cur_sep))
        possible_split = True

        for _, (_,_, sep_colors) in _cur_sep_colors.items():
            add_split = split.add(sep_colors[0], sep_colors[1])
            if not add_split:
                possible_split = False
                break
        
        if possible_split:
            out_ = []
            if print_v:
                print('\n~~~~~~~~~~~\nPOSSIBLE SEPARATOR: {}\n~~~~~~~~~~~'.format(_cur_sep))
            all_possible_splits = split.get_possible_splits()
            for possible_split in all_possible_splits:
                if print_v:
                    print('Possible_Split', possible_split)
                out_.append([_cur_sep, copy.deepcopy(_cur_sep_colors), copy.deepcopy(possible_split)])
            return out_, split
    

        # print('1:',_cur_sep, possible_split)
    # print('2:', _cur_sep, _cur_sep_possible)
    return None
                

class OutputObject:

    def __init__(self):
        # set of all aritys where a predicate exist
        self.aritys = set()

        # key = arity, value = all typecombinations with at least 1 valid predicate
        self.type_combis_for_arity = dict()

        # key = type combination, value = all possible candidates
        self.all_possible_candidates_for_typecombi = dict()

        # key = set of all possible candidates, value = all possible subsets that form an predicates
        self.valid_candidate_sets_for_typecombi = dict()

        # key = subset of candidates that form a predicate, value = possible splits for the set of candidates
        self.possible_separator_splits_valid_candidates = dict()

        self.checked_arity = set()


    def add_separator(self, type_combi, candidate, preconditions):

        if type_combi not in self.all_possible_candidates_for_typecombi:
            return None
        else:
            if candidate[0] in self.valid_candidate_sets_for_typecombi[type_combi]:
                # self.possible_separator_splits_valid_candidates[candidate[0]].append(candidate[2])
                self.possible_separator_splits_valid_candidates[candidate[0]].append([candidate[2], preconditions])
            else:
                self.valid_candidate_sets_for_typecombi[type_combi].add(candidate[0])
                self.possible_separator_splits_valid_candidates[candidate[0]] = [[candidate[2], preconditions]]

    
    def add_type_combi(self, type_combi, all_possible_candidates_for_typecombi):
        
        if len(type_combi) not in self.aritys:
            self.aritys.add(len(type_combi))
            self.type_combis_for_arity[len(type_combi)] = set()

        if type_combi not in self.type_combis_for_arity[len(type_combi)]:
            self.type_combis_for_arity[len(type_combi)].add(type_combi)
            self.all_possible_candidates_for_typecombi[type_combi] = all_possible_candidates_for_typecombi
            self.valid_candidate_sets_for_typecombi[type_combi] = set()

    def get_arities(self):
        
        arity_list = list(self.aritys)
        arity_list.sort()
        return arity_list
    
    def get_possible_separators_for_typecombi(self, type_combi):
        
        if len(type_combi) in self.type_combis_for_arity:
            if type_combi not in self.type_combis_for_arity[len(type_combi)]:
                return None
            else:
                return self.all_possible_candidates_for_typecombi[type_combi]
        else:
            return None

    def get_l_structures_for_typecombi(self, type_combi):
        
        if type_combi not in self.type_combis_for_arity[len(type_combi)]:
            return None
        else:
            return self.valid_candidate_sets_for_typecombi[type_combi]
        
    def get_typecombis_for_arity(self, arity):
        
        if arity in self.type_combis_for_arity:
            return self.type_combis_for_arity[arity]
        return None
    
    def get_splits_for_valid_separator(self, separator):
        res = []
        if separator in self.possible_separator_splits_valid_candidates:
            for [split, precondition] in self.possible_separator_splits_valid_candidates[separator]:
                res.append(split)
            return res
        return None
    
    def get_splits_for_valid_separator_with_precondition(self, separator):
        res = []
        if separator in self.possible_separator_splits_valid_candidates:
           return self.possible_separator_splits_valid_candidates[separator]
        return None
    
    def set_calculated_arity(self, arity):
        self.checked_arity.add(arity)
    
    def get_calculated_arity(self, arity):
        return arity in self.checked_arity


def get_objects_by_type(object_types):

    objects_by_type = dict()
    for _obj, _type in object_types.items():
        if _type is not None:
            if _type in objects_by_type:
                objects_by_type[_type].append(int(_obj))
            else:
                objects_by_type[_type] = [int(_obj)]

    return objects_by_type

def get_edges_by_object(G, object_mapping):

    edges_by_object = {_obj_map: set() for _, _obj_map in object_mapping.items()}
    _all_objs = {_obj_map for _, _obj_map in object_mapping.items()}
    for e in G.edges(data='action'):
        for _act in e[2]:
            for _obj in _all_objs:
                if _obj not in _act[1]:
                    edges_by_object[_obj].add((e[0],e[1]))

    return edges_by_object


def get_seperators_and_graphs_parallel(graph, locm_types, object_types, action_arity, action_list_test, action_mapping, object_mapping, pattern_groundings, num_max_worker, 
                                       prev_object: OutputObject, color_split_dict, old_benchmark_val):

    if old_benchmark_val is None:
        benchmark_val = BenchmarkValue(None)
    else:
        benchmark_val = BenchmarkValue(old_benchmark_val.get_pattern_combi())
    merged_graph_dict = {}
    output_obj = OutputObject()
    all_atoms, atom_index = dict(), 1 # dictionary for all atoms and counting varibale

    if color_split_dict is None:
        color_split_dict = {}

    # get dictionary where key=type, value=all objects of this type
    objects_by_type = get_objects_by_type(object_types)
    edges_by_object = get_edges_by_object(graph, object_mapping)

    # get max arity
    # TODO maybe better to do this over seen object combinations ???
    if prev_object is None:
        max_arity = [_ar for _ar in range(1,max(action_arity.values()) + 1)]
    else:
        max_arity = list(prev_object.get_arities())
        for _arity in range(1,max(action_arity.values()) + 1):
            if not prev_object.get_calculated_arity(_arity):
                max_arity.append(_arity)
        max_arity.sort()

    # iterate over all possible arities 
    for arity in max_arity:
        
        # for each action pattern its type tuple
        action_pattern = {(_act,_pattern):tuple([locm_types[(_act,_p)] for _p in _pattern]) for _act, _arity in action_arity.items() for _pattern in itertools.permutations(range(_arity), r=arity)}
        
        # TODO do this in a seperate fucntion
        # for each possible type tuple get all corresponding (action,pattern)
        if prev_object is None or not prev_object.get_calculated_arity(arity):
            all_type_tuple = {_type_tuple:set() for _type_tuple in set(action_pattern.values())}
            for _action_pattern, _type_tuple in action_pattern.items():
                all_type_tuple[_type_tuple].add(_action_pattern)
        else:
            all_type_tuple = {_type_tuple:set() for _type_tuple in prev_object.get_typecombis_for_arity(arity)}
            for _action_pattern, _type_tuple in action_pattern.items():
                if _type_tuple in all_type_tuple:
                    all_type_tuple[_type_tuple].add(_action_pattern)
            
        # get the number of occurences of all object tuples of the current arity
        # TODO what does action_list_test do ??
        # TODO find this out and comment this 
        occurences = count_occurences(action_list_test, object_mapping, arity, action_mapping)

        # set to store all already computed type combinations and its permutations
        allready_done = set()

        # iterate over all type combinations to find atoms for it
        for _type_combination, possible_action_pattern in all_type_tuple.items():

            benchmark_val.add_max_pattern_per_combination(len(possible_action_pattern))

            # check whether an symetric type combination was already computed
            # TODO this can be packed into a function
            if _type_combination in allready_done:
                continue
            else:
                # add all permutations of the current type combination
                for possible in itertools.permutations(list(_type_combination)):
                    allready_done.add(possible)

            benchmark_val.add_type_combinations()
            
            type_combination_possible, all_seperators = True, []
            correct_separators, separator_counter = dict(), 0

            # TODO this can be packed into a function
            # get all possible groundings based on the type combination
            _obj_list_per_position, all_types_possible = [], True
            for t in _type_combination:
                if t in objects_by_type:
                    _obj_list_per_position.append(objects_by_type[t])
                else:
                    all_types_possible = False
                    break
            
            if not all_types_possible:
                # TODO this case needs to be handeled 
                # TODO we need to add everrything that is possible for this type_combination in the previous iteration here,
                # TODO since it is here never possible because of a missing type
                # TODO Here is a BUG
                # TODO make this better, probably over the output object 
                if not prev_object is None:
                    if prev_object.get_typecombis_for_arity(arity) is None: continue
                    for _type_combi in prev_object.get_typecombis_for_arity(arity):
                        # output_obj.add_type_combi(_type_combi, )
                        if _type_combi is None: continue
                        for _separator in prev_object.get_possible_separators_for_typecombi(_type_combi):
                            if _separator is None: continue

                            output_obj.add_type_combi(_type_combi, _separator)

                            splits_and_precs = prev_object.get_splits_for_valid_separator_with_precondition(_separator)
                            if splits_and_precs is None: continue
                            for _split, _precs in splits_and_precs:
                                
                                output_obj.add_separator(_type_combi, [_separator, None, _split], _precs)
                    continue
                continue

            # [objects_by_type[t] for t in _type_combination]
            all_groundings = list(itertools.product(*_obj_list_per_position))            

            # get the grounding with the hit number
            # TODO this can be packed in a separate function
            pattern_counter = 0 
            
            # TODO pack this into a function
            grounding_occ = {_ground: 0 for _ground in all_groundings}
            for possible_act in possible_action_pattern:
                if possible_act in occurences:
                    benchmark_val.add_pattern()
                    pattern_counter += 1 
                    for ground in all_groundings:
                        if ground in occurences[possible_act]:
                            grounding_occ[ground] += occurences[possible_act][ground]
                            
            # sort the groundings in descending order
            grounding_order = sorted(grounding_occ, key=lambda k:grounding_occ[k])

            # create pool of workers to do the merging in parallel
            pool = ProcessPoolExecutor(max_workers=num_max_worker)
            res = dict()

            # currently possible action patterns
            cur_possible_action_pattern = copy.deepcopy(possible_action_pattern)
            not_possible = False

            # start for each grounding a process in which the graph is merged
            for cur_grounding in grounding_order:
                
                # print(cur_grounding)
                # don't need to merge the graph if grounding not possible
                if grounding_occ[cur_grounding] == 0:
                    continue

                # get all possible actions for this grounding
                # TODO this can be added to the parallel part
                grounding_tuple = tuple(cur_grounding)
                actions_possible_for_grounding = set()
                possible_pattern_for_grounding = set()

                # TODO this can be packed into a function
                for _action_pattern in possible_action_pattern:
                    if grounding_tuple in pattern_groundings[_action_pattern]:
                        for _possible_act in pattern_groundings[_action_pattern][grounding_tuple]:
                            actions_possible_for_grounding.add(_possible_act)
                            possible_pattern_for_grounding.add(_action_pattern)

                # if the grounding is possible then there should be a corresponding action
                if len(actions_possible_for_grounding) == 0:
                    # TODO this should not be possible
                    print('This shouldnt be possible2')
                    continue
                
                G, edges_to_delete, node_mapping, already_del_actions = None, None, None, None
                benchmark_val.add_graph()

                # TODO if one graph failed to ground we need to set the graph as NONE in the list such that for later iterations we know that we not need to ground anymore
                # TODO Need to search all smaller graphs to find the mimimal, maybe one arity was not needed after some iterations
                if arity == 1:
                    # for arity one we need to start merging on the input graph
                    G = copy_graph(graph)
                    edges_to_delete = edges_by_object[cur_grounding[0]]
                    node_mapping = {i: i for i in graph.nodes()}
                else:
                    # TODO pack this into a function 

                    # TODO need to handle the case if there object multiple times then the edge to delete is not enought since edges with one time this object are not deleted.
                    # TODO make this such that the graph is loaded
                    # TODO we probably need to catch the case such that theres is no graph of a grounding n-1, e.g. also this grounding is not possible
                    # for grounding (x_1,...,x_n) get the smallest graph for any grounding of size n-1 that is subset (x_1,...,x_n)
                    min_graph_tuple, graph_size = None, None
                    for smaller_object_tuple in itertools.permutations(cur_grounding, r=arity-1):
                        
                        # check if graph exists and get size
                        # if the graph is smaller then the current graph we choose it
                        if smaller_object_tuple in merged_graph_dict:
                            
                            if merged_graph_dict[smaller_object_tuple] is None:
                                # TODO need to close pool and do other things
                                print('Do we get here???')
                                not_possible = True
                                break

                            cur_graph_size = len(merged_graph_dict[smaller_object_tuple][0].nodes())

                            if graph_size is None:
                                min_graph_tuple = smaller_object_tuple
                                graph_size = cur_graph_size
                            elif graph_size > cur_graph_size:
                                min_graph_tuple = smaller_object_tuple
                                graph_size = cur_graph_size

                    if not not_possible:

                        if min_graph_tuple is None:
                            G = copy_graph(graph)
                            node_mapping = {_n:_n for _n in G.nodes()}
                            already_del_actions = {}

                            edges_to_delete = set()
                            for x in cur_grounding:
                                edges_to_delete.update(edges_by_object[x])
                        else:
                            # load the graph, mapping, and delered possible actions of the graph
                            load_graph, dict_node_mapping, dict_already_del_actions = merged_graph_dict[min_graph_tuple]

                            # copy the variables such that theres is no overwrite
                            node_mapping = copy.deepcopy(dict_node_mapping)
                            already_del_actions = copy.deepcopy(dict_already_del_actions)
                            G = copy_graph(load_graph)

                            # see next TODO
                            # get the object that is in the current tuple but not in the loaded graph
                            not_contained_object = [x for x in cur_grounding if x not in min_graph_tuple]

                            # get the edges of that need to be deleted for the graph
                            # TODO this only works for variables that were previosly not in the tuple, not for redundant objects
                            if len(not_contained_object) > 0:
                                edges_to_delete = edges_by_object[not_contained_object[0]]
                            else:
                                edges_to_delete = set()
                # check if we need this case   
                if not not_possible:
                    # submit the merging task to an worker
                    res[cur_grounding] = pool.submit(merge_graph_with_grounding, cur_grounding, G, node_mapping, edges_to_delete, actions_possible_for_grounding, 
                                                                                    already_del_actions, possible_action_pattern, pattern_groundings, 
                                                                                    possible_pattern_for_grounding,)
                else: 
                    break
            
            # TODO check if this can be removed 
            #if not_possible:
                # set the merged graph to None such that we know in further iterations that we don't need to continue
            #    pool.shutdown(wait=False)
            #    for cur_grounding in grounding_order:
            #        merged_graph_dict[cur_grounding] = None 
            #    continue
            

            # TODO make this prittier
            counter_finished = 0
            # get all merged graphs
            for xx in concurrent.futures.as_completed(res.values()):
                # if an merging is not successfull we can stop
                test_thing = xx.result()
                
                cur_possible_action_pattern = cur_possible_action_pattern - test_thing[2] 
                if len(cur_possible_action_pattern) == 0:
                    # print('This should die exactly here', _type_combination)
                    type_combination_possible = False
                    #pool.shutdown(wait=False)
                    #break
            
                # TODO remove, is only used for debug
                counter_finished += 1
                # TODO This can probably done with a nice progress bar, need to get number of submitted tasks 
                # print(counter_finished)

            if len(cur_possible_action_pattern) == 0:
                # print('This should die exactly here', _type_combination)
                type_combination_possible = False

            # TODO check if this can be removed 
            # if the grounding of any graph was not succsefull we do not need to check for features 
            #if not type_combination_possible:
            #    # set the merged graph to None such that we know in further iterations that we don't need to continue
            #    for cur_grounding in grounding_order:
            #        merged_graph_dict[cur_grounding] = None
            #    continue
            
            # TODO check this
            # get the merged graphs 
            for res_gr, res_value in res.items():
                res[res_gr] = res_value.result()

            for cur_grounding, [_, _, _, merged_graph] in res.items():
                merged_graph_dict[cur_grounding] = copy.deepcopy(merged_graph)  

            if not type_combination_possible:
                continue
            
            # 
            merged_graphs_for_grounding = dict()
            for grounding, [_out, _, _, _] in res.items():
                if _out is not None:
                    a,b,c,d = _out
                    merged_graphs_for_grounding[grounding] = [get_graph_from_list(a),b,c,d]
                    # print(merged_graphs_for_grounding[grounding][0])
            
            # get a list that contains the separators of each graph 
            # all_seperators = [_sep for _, _sep, _, _ in res.values()]
            all_dead_seperators = [_sep for _, _, _sep, _ in res.values()]
            possible_separators_for_grounding = {gr:_sep for gr, [_, _sep, _, _] in res.items()}

            all_seperators = set(possible_action_pattern)

            for dead_seps in all_dead_seperators:
                for d_sep in dead_seps:
                    all_seperators.discard(d_sep)


            # print(all_seperators)
            # all_seperators.append(possible_action_pattern)

            # TODO this can be removed ??? 
            if all_seperators == set():
                print('The set is empty!!!\nThis should never happen!!!')
                if prev_object is not None:
                    for _type_combi in prev_object.get_typecombis_for_arity(arity):
                        # output_obj.add_type_combi(_type_combi, )
                        if _type_combi is None: continue
                        for _separator in prev_object.get_possible_separators_for_typecombi(_type_combi):
                            if _separator is None: continue

                            output_obj.add_type_combi(_type_combi, _separator)

                            splits_and_precs = prev_object.get_splits_for_valid_separator_with_precondition(_separator)
                            if splits_and_precs is None: continue
                            for _split, _precs in splits_and_precs:
                                
                                output_obj.add_separator(_type_combi, [_separator, None, _split], _precs)
                    continue
                else:
                    # TODO what happens if for each grounding all graphs are merged into a graph that only containes 1 node?
                    print('Here is still a big problem of arity', arity)
                    continue

            # the separtor of the typecombination is the intersection of the separator of all graphs
            # TODO do also the intersection with the prev iteration if possible
            #possible_separators = set.intersection(*all_seperators)
            #possible_separators = possible_separators.intersection(cur_possible_action_pattern)

            possible_separators = all_seperators

            # print('cur_possible_action_pattern:', cur_possible_action_pattern)
            # print('possible_separators:', possible_separators)
            # print('possible_action_pattern:', possible_action_pattern)


            if prev_object is not None:
                prev_type_combi = prev_object.get_possible_separators_for_typecombi(_type_combination)
                if prev_type_combi is None:
                    continue
                possible_separators = possible_separators.intersection(prev_type_combi)

            # store the merged graphs in an dict
            # additionally reconstruct the graph (a list is returned since else there is an error)
            # merged_graphs_for_grounding = {grounding:[get_graph_from_list(a),b,c,d] for grounding, [[a,b,c,d], _, _] in res.items()}

            # create a pool to color the graph (one process for each separator set)
            pool = ProcessPoolExecutor(max_workers=num_max_worker)

            # dict to store possible candidates
            candidate = dict()

            if prev_object is None or not prev_object.get_calculated_arity(arity):
                candidate_set = powerset_without_empty_set(possible_separators)
            else:
                candidate_set = prev_object.get_l_structures_for_typecombi(_type_combination)

            # print('Action_mapping', action_mapping)
            # print('Candidates', possible_separators)
            

            # get all combinations of separators
            # TODO this can be done with the set of valid separator set in the prev iteration
            for _cur_sep in candidate_set:
                set_sep = set(_cur_sep)
                benchmark_val.add_pattern_combi(frozenset(set_sep))
                if _cur_sep in color_split_dict:
                    prev_color_split = color_split_dict[_cur_sep]
                else:
                    prev_color_split = ColorSplit(set_sep)

                # # TODO check for what the previous color_split is used 
                # for each candidate set check if all graphs are colorable
                #candidate[_cur_sep] = pool.submit(check_separator,_cur_sep, merged_graphs_for_grounding, action_mapping, object_mapping, possible_action_pattern, pattern_groundings, False,
                #                                  prev_color_split)
                #candidate[_cur_sep] = pool.submit(check_separator, _cur_sep, merged_graphs_for_grounding, action_mapping, object_mapping, possible_separators, pattern_groundings, False,
                #                                  prev_color_split)
                candidate[_cur_sep] = pool.submit(check_separator, _cur_sep, merged_graphs_for_grounding, action_mapping, object_mapping, possible_separators_for_grounding, pattern_groundings, 
                                                  False, prev_color_split)

            # wait that all colorings are finisched
            concurrent.futures.wait(candidate.values(), return_when=concurrent.futures.ALL_COMPLETED)


            # TODO make this prittier 
            candidate_counter = 0
            # check for each candidtae if the coloring was successfull
            # append that output to correct separators if the coloring is successfull
            for cur_grounding, xxx in candidate.items():
                if xxx is None:
                    print('pool exeption',xxx._exception)
                xxx = xxx.result()
                if xxx is not None:
                    
                    old_candidate_counter = candidate_counter
                    color_split_dict[cur_grounding] = xxx[1]
                    all_split_candidates = xxx[0]
                    for can in all_split_candidates:

                        benchmark_val.add_viewed_candidates()
                        # check if candidate was also possible in previous iterations
                        if prev_object is None or not prev_object.get_calculated_arity(arity):
                            if candidate_counter == 0:
                                output_obj.add_type_combi(_type_combination, possible_separators)
                            correct_separators[candidate_counter] = can
                            candidate_counter += 1
                        else:
                            prev_splits = prev_object.get_splits_for_valid_separator(can[0])
                            
                            if can[2] in prev_splits or [can[2][1],can[2][0]] in prev_splits:
                                
                                if candidate_counter == 0:
                                    output_obj.add_type_combi(_type_combination, possible_separators)

                                correct_separators[candidate_counter] = can
                                candidate_counter += 1

                    if old_candidate_counter < candidate_counter:
                        benchmark_val.add_passed_pattern()

            # TODO the whole precondition part should be in a seperate function
            correct_separator_preconditions = {}
            # check for preconditions 
            # TODO probabaly can be done in parallel, maybe not too usefull ??? 
            # check for each valid separator
            for sep_num, [_cur_sep, coloring_dict, sep_split] in correct_separators.items():

                possible_sep_precs = dict()
                

                # get colorings information for each grounding 
                for grounding, (node_color, edge_of_not_choosen_sep, separator_colors) in coloring_dict.items():
                    
                    # get node mapping and information over merged actions 
                    _, node_mapping, _, false_seps = merged_graphs_for_grounding[grounding]

                    # get possible preconditions 
                    grounding_precs = get_preconditions(_cur_sep, node_mapping, node_color, false_seps, edge_of_not_choosen_sep, sep_split, separator_colors)
                    
                    # for each not contained/not possible separator store in which color class it is possible
                    # value 3 if it is applicable in both color classes           
                    for _precondition, _precondition_value in grounding_precs.items():
                        if _precondition not in possible_sep_precs:
                            possible_sep_precs[_precondition] = _precondition_value
                        elif possible_sep_precs[_precondition] == -1:
                            possible_sep_precs[_precondition] = _precondition_value
                        elif _precondition_value == -1:
                            continue
                        elif possible_sep_precs[_precondition] != _precondition_value:
                            possible_sep_precs[_precondition] = 3
                        
            
                # add separator split together with preconditions
                if prev_object is not None and prev_object.get_calculated_arity(arity):

                    prev_splits = prev_object.get_splits_for_valid_separator(_cur_sep)
                    prev_splits_with_prec = prev_object.get_splits_for_valid_separator_with_precondition(_cur_sep)

                    if sep_split in prev_splits:
                        complement = False
                    elif [sep_split[1], sep_split[0]] in prev_splits:
                        complement = True
                    else:
                        print('Here is a new Bug')
                        


                    if complement:
                        
                        prev_pre = None

                        for split, prec in prev_splits_with_prec:
                            if split == [sep_split[1], sep_split[0]]:
                                prev_pre = prec
                                break
                        

                        #print('COMPLEMENT: {} \nPREV_PRE: {}\nPREV_NOW: {}'.format(complement,prev_pre, possible_sep_precs))

                        for _precondition, _precondition_value in list(possible_sep_precs.items()):
                            if _precondition in prev_pre:
                                if _precondition_value == 3:
                                    continue
                                elif prev_pre[_precondition] == 3:
                                    possible_sep_precs[_precondition] = 3
                                elif prev_pre[_precondition] == -1:
                                    continue 
                                elif possible_sep_precs[_precondition] == -1:
                                    possible_sep_precs[_precondition] = 1 - prev_pre[_precondition]
                                elif prev_pre[_precondition] + possible_sep_precs[_precondition] != 1:
                                    # print(_precondition_value , possible_sep_precs[_precondition], _precondition_value + possible_sep_precs[_precondition])
                                    possible_sep_precs[_precondition] = 3
                        #print('PREC_AFTER: ', possible_sep_precs)
                    else:

                        prev_pre = None

                        for split, prec in prev_splits_with_prec:
                            if split == sep_split:
                                prev_pre = prec
                                break
                        
                        #print('PREV_PRE: {}\nPREV_NOW: {}'.format(prev_pre, possible_sep_precs))

                        for _precondition, _precondition_value in list(possible_sep_precs.items()):
                            if _precondition in prev_pre:
                                if _precondition_value == 3:
                                    continue
                                elif prev_pre[_precondition] == 3:
                                    possible_sep_precs[_precondition] = 3
                                elif prev_pre[_precondition] == -1 or possible_sep_precs[_precondition] == -1:
                                    possible_sep_precs[_precondition] = max([prev_pre[_precondition],possible_sep_precs[_precondition]])
                                elif prev_pre[_precondition] != possible_sep_precs[_precondition]:
                                    possible_sep_precs[_precondition] = 3

                        #print('PREC_AFTER: ', possible_sep_precs)



                output_obj.add_separator(_type_combination, [_cur_sep, coloring_dict, sep_split], possible_sep_precs)
                correct_separator_preconditions[sep_num] = possible_sep_precs

            output_obj.set_calculated_arity(arity)                
            # add all valid separators to the output and increase number of atoms
            for sep_num, [_cur_sep, _, _sep_split] in correct_separators.items():
                all_atoms[atom_index] = [_cur_sep, _sep_split, correct_separator_preconditions[sep_num]]
                atom_index += 1
    '''
    for arity in output_obj.get_arities():
        num_sepes = 0
        test0 = output_obj.get_typecombis_for_arity(arity) 
        if test0 is None: continue
        for type_combi in output_obj.get_typecombis_for_arity(arity):
            test1 = output_obj.get_l_structures_for_typecombi(type_combi)
            if test1 is None: continue
            for sep in output_obj.get_l_structures_for_typecombi(type_combi):
                test2 = output_obj.get_splits_for_valid_separator_with_precondition(sep)
                if test2 is None: continue
                print(len(output_obj.get_splits_for_valid_separator_with_precondition(sep)))
                num_sepes += len(output_obj.get_splits_for_valid_separator_with_precondition(sep))
        print('ARITY {} has {} predicates'.format(arity, num_sepes))
        '''
    
    # TODO this should be handeled in a different way
    for i in range(1,max(action_arity.values()) + 1):
        if prev_object is not None and prev_object.get_calculated_arity(i):
            output_obj.set_calculated_arity(i)

    print(benchmark_val)
    #print('Calculated Arities', output_obj.get_arities())
    #for i in range(1,max(action_arity.values()) + 1):
    #    print(i, output_obj.get_calculated_arity(i))
    # return all valid predicates

    return all_atoms, output_obj, color_split_dict, benchmark_val


def get_all_type_tuple(output_object, arity, action_pattern):
    all_type_tuple = {_type_tuple:set() for _type_tuple in output_object.get_typecombis_for_arity(arity)}
    for _action_pattern, _type_tuple in action_pattern.items():
        if _type_tuple in all_type_tuple:
            all_type_tuple[_type_tuple].add(_action_pattern)
    
    return all_type_tuple


def get_grounding_occ(all_groundings, possible_action_pattern, occurences):
    grounding_occ = {_ground: 0 for _ground in all_groundings}
    for possible_act in possible_action_pattern:
        for ground in all_groundings:
            if possible_act in occurences:
                if ground in occurences[possible_act]:
                    grounding_occ[ground] += occurences[possible_act][ground]
    
    return grounding_occ

def get_possibilities_actions_and_pattterns(possible_action_pattern, grounding_tuple, pattern_groundings):    
    actions_possible_for_grounding, possible_pattern_for_grounding = set(), set()
    for _action_pattern in possible_action_pattern:
        if grounding_tuple in pattern_groundings[_action_pattern]:
            for _possible_act in pattern_groundings[_action_pattern][grounding_tuple]:
                actions_possible_for_grounding.add(_possible_act)
            possible_pattern_for_grounding.add(_action_pattern)
    
    return actions_possible_for_grounding, possible_pattern_for_grounding

def verify_parallel(output_object:OutputObject, in_graph, locm_types, object_types, action_arity, action_list_test, action_mapping, object_mapping, pattern_groundings,
                    num_max_worker, verification_mode, verification_truth_value):
    
    verification_value, merged_graph_dict = True, {}

    # get dictionary where key=type, value=all objects of this type
    objects_by_type = get_objects_by_type(object_types)
    edges_by_object = get_edges_by_object(in_graph, object_mapping)

    # dictionary for all atoms and counting varibale
    all_atoms, atom_index = dict(), 1

    # iterate over all possible arities 
    for arity in output_object.get_arities():

        # for each action pattern its type tuple 
        action_pattern = {(_act,_pattern):tuple([locm_types[(_act,_p)] for _p in _pattern]) for _act, _arity in action_arity.items() for _pattern in itertools.permutations(range(_arity), r=arity)}

        # for each possible type tuple get all corresponding (action,pattern)       
        all_type_tuple = get_all_type_tuple(output_object, arity, action_pattern)

        # get the number of occurences of all object tuples of the current arity
        # TODO what does action_list_test do ??
        occurences = count_occurences(action_list_test, object_mapping, arity, action_mapping)

        # set to store all already computed type combinations and its permutations
        allready_done = set()

        # iterate over all type combinations to find atoms for it
        for _type_combination in output_object.get_typecombis_for_arity(arity):
            
            possible_action_pattern = all_type_tuple[_type_combination]

            # check whether an symetric type combination was already computed
            if _type_combination in allready_done:
                continue
            else:
                # add all permutations of the current type combination
                for possible in itertools.permutations(list(_type_combination)):
                    allready_done.add(possible)
            
            type_combination_possible, all_seperators, correct_separators, separator_counter = True, [], dict(), 0

            # get all possible groundings based on the type combination
            # There is the possibility that there is not object of a type in the verification instance
            # threfore we assume that the predicates for the type_combination are possible
            try:
                _obj_list_per_position = [objects_by_type[t] for t in _type_combination]
            except KeyError:
                continue

            all_groundings = list(itertools.product(*_obj_list_per_position))            

            # get the grounding with the maximal hit number
            # TODO this can be packed in a separate function
            grounding_occ = get_grounding_occ(all_groundings, possible_action_pattern, occurences)

            # sort the groundings in descending order
            grounding_order = sorted(grounding_occ, key=lambda k:grounding_occ[k])

            # create pool of workers to do the merging in parallel
            pool = ProcessPoolExecutor(max_workers=num_max_worker)
            res, not_possible = dict(), False

            # currently possible action patterns
            cur_possible_action_pattern = copy.deepcopy(possible_action_pattern)

            # start for each grounding a process in which the graph is merged
            for cur_grounding in grounding_order:
                
                # print(cur_grounding)
                # don't need to merge the graph if grounding not possible
                if grounding_occ[cur_grounding] == 0:
                    continue

                # get all possible actions for this grounding
                grounding_tuple = tuple(cur_grounding)

                actions_possible_for_grounding, possible_pattern_for_grounding = get_possibilities_actions_and_pattterns(possible_action_pattern, grounding_tuple, pattern_groundings)

                # if the grounding is possible then there should be a corresponding action
                if len(actions_possible_for_grounding) == 0:
                    # TODO this should not be possible
                    print('This shouldnt be possible3')
                    continue
                
                G, edges_to_delete, node_mapping, already_del_actions = None, None, None, None

                if arity == 1:
                    # for arity one we need to start merging on the input graph
                    G = copy_graph(in_graph)
                    edges_to_delete = set()
                    for _obj in cur_grounding:
                        for _edge in edges_by_object[_obj]:
                            edges_to_delete.add(_edge)
                    node_mapping = {i: i for i in in_graph.nodes()}
                else:
                    # TODO maybe it needs to be possible to load the initial graph if there is no predicate of arity 0.
                    # TODO make this such that the graph is loaded
                    # TODO we probably need to catch the case such that theres is no graph of a grounding n-1, e.g. also this grounding is not possible
                    # for grounding (x_1,...,x_n) get the smallest graph for any grounding of size n-1 that is subset (x_1,...,x_n)
                    min_graph_tuple, graph_size = None, None
                    for i in range(1,arity):
                        for smaller_object_tuple in itertools.permutations(cur_grounding, r=i):
                            
                            # check if graph exists and get size
                            # if the graph is smaller then the current graph we choose it
                            if smaller_object_tuple in merged_graph_dict:
                                
                                if merged_graph_dict[smaller_object_tuple] is None:
                                    # TODO need to close pool and do other things
                                    not_possible = True
                                    break

                                cur_graph_size = len(merged_graph_dict[smaller_object_tuple][0].nodes())

                                if graph_size is None:
                                    min_graph_tuple = smaller_object_tuple
                                    graph_size = cur_graph_size
                                elif graph_size > cur_graph_size:
                                    min_graph_tuple = smaller_object_tuple
                                    graph_size = cur_graph_size

                    if not not_possible:
                        # load the graph, mapping, and delered possible actions of the graph
                        if min_graph_tuple is None:
                            
                            G = copy_graph(in_graph)
                            node_mapping = {_n:_n for _n in G.nodes()}
                            already_del_actions = {}

                            edges_to_delete = set()
                            for x in cur_grounding:
                                edges_to_delete.update(edges_by_object[x])

                        else:
                            graph, dict_node_mapping, dict_already_del_actions = merged_graph_dict[min_graph_tuple]

                            # copy the variables such that theres is no overwrite
                            node_mapping, already_del_actions, G = copy.deepcopy(dict_node_mapping), copy.deepcopy(dict_already_del_actions), copy_graph(graph)

                            # get the object that is in the current tuple but not in the loaded graph
                            not_contained_object = [x for x in cur_grounding if x not in min_graph_tuple]

                            # get the edges of that need to be deleted for the graph
                            # TODO this only works for variables that were previosly not in the tuple, not for redundant objects
                            edges_to_delete = set()
                            for x in not_contained_object:
                                edges_to_delete.update(edges_by_object[x])


                if not not_possible:
                    # submit the merging task to an worker
                    res[cur_grounding] = pool.submit(merge_graph_with_grounding, cur_grounding, G, node_mapping, edges_to_delete, actions_possible_for_grounding, 
                                                                            already_del_actions, possible_action_pattern, pattern_groundings, possible_pattern_for_grounding)
                else:
                    break
            
            
            #if not_possible:
                # set the merged graph to None such that we know in further iterations that we don't need to continue
            #    pool.shutdown(wait=False)
            #    for cur_grounding in grounding_order:
            #        merged_graph_dict[cur_grounding] = None 
            #    continue

            counter_finished = 0

            # get all merged graphs
            for xx in concurrent.futures.as_completed(res.values()):
                # if an merging is not successfull we can stop

                test_thing = xx.result()
                
                cur_possible_action_pattern = cur_possible_action_pattern - test_thing[2] 
                if len(cur_possible_action_pattern) == 0:
                    type_combination_possible = False
                    pool.shutdown(wait=False)
                    # print('Was the type combination possible ? ')
                
                counter_finished += 1

            # get results
            for res_gr, res_value in res.items():
                parallel_val = res_value.result()
                if parallel_val is not None:
                    res[res_gr] = parallel_val

            # save the merged graph such that they can be used in the next arity
            for cur_grounding, [_, _, _, merged_graph] in res.items():
                    merged_graph_dict[cur_grounding] = merged_graph
                    
            # if all patterns are not possible we can continue
            # TODO i'm not shure if we can set the graphs to None or we have to search further ??? 
            if not type_combination_possible:
                #for cur_grounding in grounding_order:
                #    merged_graph_dict[cur_grounding] = None
                continue
            
            # get a list that contains the separators of each graph and add all separators possible
            all_dead_seperators = [_sep for _, _, _sep, _ in res.values()]
            possible_separators_for_grounding = {gr:_sep for gr, [_, _sep, _, _] in res.items()}
            all_seperators = set(possible_action_pattern)

            for dead_seps in all_dead_seperators:
                for d_sep in dead_seps:
                    all_seperators.discard(d_sep)

            possible_separators = all_seperators
            # the separtor of the typecombination is the intersection of the separator of all graphs
            # possible_separators = set.intersection(*all_seperators)

            # store the merged graphs in an dict
            # additionally reconstruct the graph (a list is returned since else there is an error)
            merged_graphs_for_grounding = dict()
            for grounding, [_out, _, _, _] in res.items():
                if _out is not None:
                    a,b,c,d = _out
                    merged_graphs_for_grounding[grounding] = [get_graph_from_list(a),b,c,d]

            # create a pool to color the graph (one process for each separator set)
            pool = ProcessPoolExecutor(max_workers=num_max_worker)

            # dict to store possible candidates
            candidate = dict()

            # get all combinations of separators
            for _cur_sep in output_object.get_l_structures_for_typecombi(_type_combination):
                # for each candidate set check if all graphs are colorable
                # TODO we need to use the split here
                # TODO probabaly need to use a fixed coloring
                # TODO we can also use just the separator and then check whether all splits are found
                _sep_set = set(_cur_sep)
                color_split = ColorSplit(_sep_set)
                candidate[_cur_sep] = pool.submit(check_separator,_cur_sep, merged_graphs_for_grounding, action_mapping, object_mapping, possible_separators_for_grounding, 
                                                  pattern_groundings, False, color_split)

            # wait that all colorings are finisched
            concurrent.futures.wait(candidate.values(), return_when=concurrent.futures.ALL_COMPLETED)

            candidate_counter = 0
            # check for each candidtae if the coloring was successfull
            # append that output to correct separators if the coloring is successfull
            for cur_separator, xxx in candidate.items():
                # TODO get the possible splits for the separator
                # TODO then check uf all of them are found in the current verification instance
                xxx = xxx.result()
                if xxx is not None:
                    prev_possible_splits = copy.deepcopy(output_object.get_splits_for_valid_separator(cur_separator))
                    debug_split_list = []
                    positions_to_delete = []

                    for can in xxx[0]:
                        split = can[2]
                        debug_split_list.append(split)
                        for position, prev_candidate in enumerate(list(prev_possible_splits)):
                            if split == prev_candidate:
                                correct_separators[candidate_counter] = can
                                candidate_counter += 1
                                positions_to_delete.append(position)
                            elif [split[1],split[0]] == prev_candidate:
                                correct_separators[candidate_counter] = can
                                candidate_counter += 1
                                positions_to_delete.append(position)

                    positions_to_delete.sort(reverse=True)
                    for position in positions_to_delete:
                        del prev_possible_splits[position]

                    if len(prev_possible_splits) > 0:
                        # print('For separator {} the following splits were not found:'.format(cur_separator))
                        #for num_prev_split, prev_split in enumerate(prev_possible_splits):
                        #    print(num_prev_split+1, '.   ', prev_split[0], '     ', prev_split[1])
                        #print('\n')
                        print('HERE IT BREAKS 1')
                        verification_value = False
                else:
                    # print('This separator is not possible in the verification instance: {} \n'.format(cur_separator))
                    print('HERE IT BREAKS 2')
                    verification_value = False
            
            if not verification_value and verification_mode and verification_truth_value:
                return verification_value
            
            if not verification_value and not verification_truth_value:
                return not verification_value

            # TODO HERE NEEDS TO BE A CHECK OF SEPARATORS
            correct_separator_preconditions = {}

            # check for preconditions
            # check for each valid separator
            for sep_num, [_cur_sep, coloring_dict, sep_split] in correct_separators.items():

                possible_sep_precs = dict()
                
                # get colorings information for each grounding 
                for grounding, (node_color, edge_of_not_choosen_sep, separator_colors) in coloring_dict.items():
                    
                    # get node mapping and information over merged actions 
                    _, node_mapping, _, false_seps = merged_graphs_for_grounding[grounding]

                    # get possible preconditions 
                    grounding_precs = get_preconditions(_cur_sep, node_mapping, node_color, false_seps, edge_of_not_choosen_sep, sep_split, separator_colors)
                    
                    #print(grounding_precs)
                    #print(false_seps.keys())
                    #print(edge_of_not_choosen_sep.keys())

                    # for each not contained/not possible separator store in which color class it is possible
                    # value 3 if it is applicable in both color classes
                    for _precondition, _precondition_value in grounding_precs.items():
                        if _precondition not in possible_sep_precs:
                            possible_sep_precs[_precondition] = _precondition_value
                        elif possible_sep_precs[_precondition] == -1:
                            possible_sep_precs[_precondition] = _precondition_value
                        elif _precondition_value == -1:
                            continue
                        elif possible_sep_precs[_precondition] != _precondition_value:
                            possible_sep_precs[_precondition] = 3
                    
                # TODO this should probabaly be moved one position to the left
                # save dict with precondition value for all separators that are not contained
                correct_separator_preconditions[sep_num] = possible_sep_precs
    
                prev_possible_splits_with_prec = output_object.get_splits_for_valid_separator_with_precondition(_cur_sep)
                
                is_complement, _split_preconditions, preconditions_possible = None, None, True

                for prev_possible in prev_possible_splits_with_prec:
                    if prev_possible[0] == sep_split:
                        is_complement = False
                        _split_preconditions = prev_possible[1]
                        break
                    elif prev_possible[0] == [sep_split[1], sep_split[0]]:
                        is_complement = True
                        _split_preconditions = prev_possible[1]
                        break

                if is_complement == None:
                    print('This shouldnt be possible4')
                    return None

                if not is_complement:

                    for prec, prec_val in _split_preconditions.items():
                        if prec in possible_sep_precs:
                            new_val = possible_sep_precs[prec]
                            
                            if new_val == -1:
                                continue
                            elif prec_val == -1 and verification_truth_value:
                                print('HERE IT BREAKS 3')
                                verification_value = False
                                if preconditions_possible:
                                    preconditions_possible = False
                                #    print('For separator {} for the split {} the following preconditions are violated:'.format(_cur_sep, sep_split))
                                #print('Action: {}   Learned Precondition = {}   Verification Precondition = {}'.format(prec, prec_val, new_val))
                            elif prec_val == 3:
                                continue   
                            elif new_val != prec_val and prec_val != -1:
                                print('HERE IT BREAKS 4')
                                verification_value = False
                                if preconditions_possible:
                                    preconditions_possible = False
                                #    print('For separator {} for the split {} the following preconditions are violated:'.format(_cur_sep, sep_split))
                                #print('Action: {}   Learned Precondition = {}   Verification Precondition = {}'.format(prec, prec_val, new_val))
                        #else:
                        #    print('HERE IT BREAKS 5')
                        #    print(prec)
                        #    print(_cur_sep)
                        #    print(_split_preconditions)
                        #    print(possible_sep_precs)
                        #    print(all_dead_seperators)
                        #    verification_value = False
                        #    if preconditions_possible:
                        #        preconditions_possible = False
                                #print('For separator {} for the split {} the following preconditions are violated:'.format(_cur_sep, sep_split))
                            #print('Action: {}   Learned Precondition = {}   Verification Precondition = {}'.format(prec, prec_val, new_val))
                else:
                    # TODO here the preconditions in the output should be "complement for one side"
                    for prec, prec_val in _split_preconditions.items():
                        if prec in possible_sep_precs:
                            new_val = possible_sep_precs[prec]

                            # This is only needed to print the value right
                            if 0 <= prec_val <= 1:
                                output_val = 1-prec_val
                            else: 
                                output_val = prec_val

                            if new_val == -1:
                                continue
                            elif prec_val == -1 and verification_truth_value:
                                print('HERE IT BREAKS 6')
                                if preconditions_possible:
                                    preconditions_possible = False
                                    #print('For separator {} for the split {} the following preconditions are violated:'.format(_cur_sep, sep_split))
                                verification_value = False
                                #print('Action: {}   Learned Precondition = {}   Verification Precondition = {}'.format(prec, output_val, new_val))
                            elif new_val + prec_val == 1 and prec_val > -1:
                                continue
                            elif prec_val == 3:
                                continue
                            elif prec_val > -1:
                                print('HERE IT BREAKS 7')
                                verification_value = False
                                if preconditions_possible:
                                    preconditions_possible = False
                                    #print('For separator {} for the split {} the following preconditions are violated:'.format(_cur_sep, sep_split))
                                #print('Action: {}   Learned Precondition = {}   Verification Precondition = {}'.format(prec, output_val, new_val))
                        #else:
                        #    print('HERE IT BREAKS 8')
                        #    print(prec)
                        #    print(_cur_sep)
                        #    print(_split_preconditions)
                        #    print(possible_sep_precs)
                        #    print(all_dead_seperators)
                        #    verification_value = False
                        #    if preconditions_possible:
                        #        preconditions_possible = False
                                #print('For separator {} for the split {} the following preconditions are violated:'.format(_cur_sep, sep_split))
                            #print('Action: {}   Learned Precondition = {}   Verification Precondition = {}'.format(prec, output_val, new_val))

                #if not preconditions_possible:
                    #print('\n')

            if not verification_value and verification_mode and verification_truth_value:
                return verification_value
            elif not verification_value and not verification_truth_value:
                return not verification_value

    # return all valid predicates
    if verification_truth_value:
        return verification_value
    else:
        print('HERE ITS FALSE')
        return not verification_value

# BFS like coloring 
def get_coloring(G, grounding, contained_seps, action_mapping, object_mapping, all_separators, pattern_grounding):

    not_used_seps = set(all_separators)-set(contained_seps)

    # edges of not choosen separators
    edge_of_not_choosen_sep = {x: set() for x in not_used_seps}

    # color of a node 
    node_color = {i: None for i in G.nodes()}

    # color in which a separator starts
    sep_colors = {_sep: None for _sep in contained_seps}

    # use a random node as starting point and set its color
    initial_node = list(G.nodes())[0]
    node_color[initial_node] = 0

    # list that contains node where the color got determined but the node was not visited
    set_color = [initial_node]

    # all nodes that were already visited
    alredy_visited = set()

    # boolean that states whether it is a valid coloring
    valid = True

    seen = set()
                        
    # TODO can probably done better currently see each edge twice
    while len(set_color) > 0 and valid:

        # get node with color that was not visited before
        node = set_color.pop(0)
        
        # get all outgoing edges for the node
        for edge in G.out_edges([node],data='action'):
            # only check nodes with larger index
            #if edge[0] < edge[1]:
            if (edge[0], edge[1]) not in seen:

                seen.add((edge[0], edge[1]))

                # get all separators that fit the edge
                seps_for_grounding = set()
                for _sep in contained_seps:
                    for _cur_action in edge[2]:
                        if grounding in pattern_grounding[_sep]:
                            if _cur_action in pattern_grounding[_sep][grounding]:
                                seps_for_grounding.add(_sep)
                                break
                
                all_grounding_seps = set()
                for _sep in contained_seps:
                    if grounding in pattern_grounding[_sep]:
                        for _cur_action in edge[2]:
                            if _cur_action in pattern_grounding[_sep][grounding]:
                                all_grounding_seps.add(_cur_action)
                                
                
                if len(seps_for_grounding) > 0: 
                    if len(all_grounding_seps) < len(edge[2]):
                        return None

                # if there is some the color has to switch
                if len(seps_for_grounding) > 0:
                    # set color if not already
                    if node_color[edge[1]] is None:
                        # set node color
                        node_color[edge[1]] = 1 - node_color[edge[0]]
                    # check if correctly colored
                    elif node_color[edge[1]] + node_color[edge[0]] != 1:
                        # no valid separator
                        valid = False
                        #print('HERE IT BREAKS 1')
                        return None
                    
                    # check if all separators start in the correct color
                    for _seps in seps_for_grounding:
                        # set color if not set
                        if sep_colors[_seps] is None:
                            sep_colors[_seps] = node_color[edge[0]]
                        # break if separator starts in wrong color
                        elif sep_colors[_seps] != node_color[edge[0]]:
                            # no valid separator 
                            valid = False
                            #print('HERE IT BREAKS 2')
                            return None
                # if no separator fits the edge nodes need to be colored the same
                else:
                    # color not if not already
                    if node_color[edge[1]] is None:
                        # set node color
                        node_color[edge[1]] = node_color[edge[0]]
                    # check coloring, break if not correct
                    elif node_color[edge[1]] != node_color[edge[0]]:
                        # no valid separator
                        valid = False
                        #print('HERE IT BREAKS 3')
                        return None
                fitting_not_used_seps = set()
                for _sep in not_used_seps:
                    for _cur_action in edge[2]:
                        if grounding in pattern_grounding[_sep]:
                            if _cur_action in pattern_grounding[_sep][grounding]:
                                fitting_not_used_seps.add(_sep)
                if len(fitting_not_used_seps) > 0 and len(seps_for_grounding) > 0:
                   # print('Here it breaks 8')
                    return None
                for _not_used_sep in fitting_not_used_seps:
                    edge_of_not_choosen_sep[_not_used_sep].add((edge[0],edge[1]))
                    # add node to queue if not already contained or already visited
                if edge[1] not in alredy_visited:
                    if edge[1] not in set_color: 
                        set_color.append(edge[1])
        
        # check all ingoing edges
        for edge in G.in_edges([node],data='action'):

            # only check nodes with larger index 
            if (edge[0], edge[1]) not in seen:

                seen.add((edge[0], edge[1]))

                # get all separators that fit the edge
                #seps_for_grounding = separatores_fit_grounded_action(grounding, contained_seps, edge[2], action_mapping, object_mapping)
                seps_for_grounding = set()
                for _sep in contained_seps:
                    for _cur_action in edge[2]:
                        if grounding in pattern_grounding[_sep]:
                            if _cur_action in pattern_grounding[_sep][grounding]:
                                seps_for_grounding.add(_sep)
                                break

                all_grounding_seps = set()
                for _sep in contained_seps:
                    if grounding in pattern_grounding[_sep]:
                        for _cur_action in edge[2]:
                            if _cur_action in pattern_grounding[_sep][grounding]:
                                all_grounding_seps.add(_cur_action)
                                
                
                if len(seps_for_grounding) > 0: 
                    if len(all_grounding_seps) < len(edge[2]):
                        return None        

                # if there is some the color has to switch
                if len(seps_for_grounding) > 0:
                    # set color if not already
                    if node_color[edge[0]] is None:
                        # set node color
                        node_color[edge[0]] = 1 - node_color[edge[1]]
                    # check if correctly colored
                    elif node_color[edge[1]] + node_color[edge[0]] != 1:
                        # no valid separator
                        valid = False
                       # print('HERE IT BREAKS 4')
                        # set_color = set()
                        return None
                    # check if all separators start in the correct color
                    for _seps in seps_for_grounding:
                        # set color if not set
                        if sep_colors[_seps] is None:
                            sep_colors[_seps] = node_color[edge[0]]
                        # break if separator starts in wrong color
                        elif sep_colors[_seps] != node_color[edge[0]]:
                            # no valid separator
                            valid = False
                          #  print('HERE IT BREAKS 5')
                            return None
                else:
                    # color not if not already
                    if node_color[edge[0]] is None:
                        # set node color
                        node_color[edge[0]] = node_color[edge[1]]
                    # check coloring, break if not correct
                    elif node_color[edge[0]] != node_color[edge[1]]:
                        # no valid separator
                        #print('HERE IT BREAKS 6')
                        set_color = set()
                        valid = False
                        return None
                fitting_not_used_seps = set()
                for _sep in not_used_seps:
                    for _cur_action in edge[2]:
                        if grounding in pattern_grounding[_sep]:
                            if _cur_action in pattern_grounding[_sep][grounding]:
                                fitting_not_used_seps.add(_sep)
                if len(fitting_not_used_seps) > 0 and len(seps_for_grounding) > 0:
                   # print('Here it breaks 7')
                    return None

                for _not_used_sep in fitting_not_used_seps:
                    edge_of_not_choosen_sep[_not_used_sep].add((edge[0],edge[1]))

                # add node to queue if not already contained or already visited
                if edge[0] not in alredy_visited:
                    if edge[0] not in set_color: 
                        set_color.append(edge[0])
        
        # add current node to visited
        alredy_visited.add(node)

    if valid:
        # TODO this is not good 
        separator_colors = [set(), set()]
        for _sep, _sep_col in sep_colors.items():
            if _sep_col is not None:
                separator_colors[_sep_col].add(_sep)
        return node_color, edge_of_not_choosen_sep, separator_colors
    else:
        return None


# Here we assume that the add/delete list is already set, e.g. there is a unique split between add and delete
def get_preconditions(cur_sep, node_mapping, node_color, false_seps, edge_of_not_choosen_sep, split, separator_color):

    possible_precs = {}

    for f_sep, edge in false_seps.items():

        c_class = None

        for e in edge:

            start = node_mapping[e[0]]
            while node_mapping[start] != start:
                start = node_mapping[start]

            if c_class is None:
                c_class = node_color[start]
            elif c_class != node_color[start]:
                c_class = 3
                break
        
        if c_class == 0:
            if separator_color[0] == set() and separator_color[1] == set():
                possible_precs[f_sep] = -1
            elif split[0] is None or split[1] is None:
                possible_precs[f_sep] = -1
            elif separator_color[0].issubset(split[0]) and separator_color[1].issubset(split[1]):
                possible_precs[f_sep] = 0
            elif separator_color[1].issubset(split[0]) and separator_color[0].issubset(split[1]):
                possible_precs[f_sep] = 1
            else:
                print('This does not really work...')
        elif c_class == 1:
            if separator_color[0] == set() and separator_color[1] == set():
                possible_precs[f_sep] = -1
            elif split[0] is None or split[1] is None:
                possible_precs[f_sep] = -1
            elif separator_color[0].issubset(split[0]) and separator_color[1].issubset(split[1]):
                possible_precs[f_sep] = 1
            elif separator_color[1].issubset(split[0]) and separator_color[0].issubset(split[1]):
                possible_precs[f_sep] = 0
            else:
                print('This does not really work...')

        elif c_class is None:
            possible_precs[f_sep] = -1
        elif c_class == 3:
            possible_precs[f_sep] = 3

        
        
    for nc_sep, edge in edge_of_not_choosen_sep.items():

        c_class = None
        
        for e in edge:
            
            start = e[0]
            while node_mapping[start] != start:
                start = node_mapping[start]

            if c_class is None:
                c_class = node_color[start]
            elif c_class != node_color[start]:
                c_class = 3
                break


        if c_class == 0:
            if separator_color[0] == set() and separator_color[1] == set():
                possible_precs[nc_sep] = -1
            elif split[0] is None or split[1] is None:
                possible_precs[nc_sep] = -1
            elif separator_color[0].issubset(split[0]) and separator_color[1].issubset(split[1]):
                possible_precs[nc_sep] = 0
            elif separator_color[1].issubset(split[0]) and separator_color[0].issubset(split[1]):
                possible_precs[nc_sep] = 1
            else:
                print('This does not really work...')
        elif c_class == 1:
            if separator_color[0] == set() and separator_color[1] == set():
                possible_precs[nc_sep] = -1
            elif split[0] is None or split[1] is None:
                possible_precs[nc_sep] = -1
            elif separator_color[0].issubset(split[0]) and separator_color[1].issubset(split[1]):
                possible_precs[nc_sep] = 1
            elif separator_color[1].issubset(split[0]) and separator_color[0].issubset(split[1]):
                possible_precs[nc_sep] = 0
            else:
                print('This does not really work...')

        elif c_class is None:
            possible_precs[nc_sep] = -1
        elif c_class == 3:
            possible_precs[nc_sep] = 3
    
    return possible_precs


def get_pattern_groundings(action_arity, all_actions):

    pattern_groundings = {(_action, _pattern):dict() for _action, _arity in action_arity.items() for _i in range(1,_arity+1) for _pattern in itertools.permutations(range(_arity), r=_i)}

    for (action,objects) in all_actions:
        for i in range(1, len(objects)+1):
            for pattern in itertools.permutations(range(len(objects)), r=i):
                _object_tuple = tuple([objects[_pos] for _pos in pattern])
                if _object_tuple in pattern_groundings[(action, pattern)]:
                    pattern_groundings[(action, pattern)][_object_tuple].add((action, objects))
                else:
                    pattern_groundings[(action, pattern)][_object_tuple] = {(action,objects)}

    return pattern_groundings

    
def count_action_groundings_new(graph, all_actions):

    counted_actions = {_act: 0 for _act in all_actions}

    for edge in graph.edges(data='action'):
        for _act in edge[2]:
            counted_actions[_act] += 1

    return counted_actions



# TODO here we assume that each action is contained in each instance, therefore there can not be an action name which is in a separator and its 
# TODO need to take care that candidates that have never been seen should not be deleted, prob. use a list that contains already seen actions
# TODO especially needed for action sets that have never been seen before, each subset of them needs to be take into account.
# color split (add / delete list) is not unambigous
def get_zeronary_features(G_full, action_mapping, prev_features, old_benchmark_val):

    if old_benchmark_val is None:
        benchmark_val = BenchmarkValue(None)
    else:
        benchmark_val = BenchmarkValue(old_benchmark_val.get_pattern_combi())

    # get graph for zeronary features
    G_zero, possible_actions = get_nx_graph_only_action_names_new(G_full)

    print(possible_actions)

    # get all possible actions
    all_action_names = {_act for _act in action_mapping.values()}

    possible_zeronary, zeronary_features = dict(), dict()

    if prev_features is None:
        feature_candidates = powerset_without_empty_set(all_action_names)
    else:
        feature_candidates = prev_features.keys()
    
    possible_preconditions = dict()

    num_candidates = 0

    # test all colorings
    for _cur_sep in feature_candidates:

        num_candidates += 1

        sep_set = set(_cur_sep)

        benchmark_val.add_pattern_combi(frozenset(sep_set))

        coloring = get_zeronary_coloring(G_zero, sep_set, possible_actions)

        if coloring is not None:
            if prev_features is not None:
                prev_color_split = prev_features[_cur_sep][0]
                coloring_possible = prev_color_split.add(coloring[2][0], coloring[2][1])
                if not coloring_possible:
                    continue
                else:
                    zeronary_features[_cur_sep] = prev_color_split
            else:
                cs = ColorSplit(sep_set)
                cs.add(coloring[2][0], coloring[2][1])
                zeronary_features[_cur_sep] = cs
            
            
            for split in zeronary_features[_cur_sep].get_possible_splits():
                if _cur_sep in possible_zeronary:
                    possible_zeronary[_cur_sep].append([coloring, split])
                else:
                    possible_zeronary[_cur_sep] = [[coloring, split]]
        
    num_splits = 0
    for zeronay, split_list in possible_zeronary.items():
        benchmark_val.add_passed_pattern()
        for zeronary_coloring, zeronary_split in split_list:
            
            num_splits += 1

            node_coloring = zeronary_coloring[0]
            edge_of_not_choosen_sep = zeronary_coloring[1]
            current_split = zeronary_coloring[2]

            possible_preconditions_for_split = {}

            # TODO add complement
            prev_precs, can_continue, need_complement = None, False, False
            if prev_features is not None:
                for prev_split, prev_split_precs in prev_features[zeronay][1]:
                    
                    if current_split[0] == set() and current_split[1] == set():
                        prev_precs = prev_split_precs
                        can_continue = True
                        break
                    elif current_split[0].issubset(prev_split[0]) and current_split[1].issubset(prev_split[1]):
                        prev_precs = prev_split_precs
                        break
                    elif current_split[1].issubset(prev_split[0]) and current_split[1].issubset(prev_split[0]):
                        prev_precs = copy.deepcopy(prev_split_precs)
                        for prec in prev_precs:
                            if 0 <= prev_precs[prec] <= 1:
                                prev_precs[prec] = 1 - prev_precs[prec]
                        need_complement = True
                        break
                    
                if prev_precs is None:
                    print('This should not happen!')
                    break
            
            #if can_continue:
            #    continue

            mult_color = False
            for value in node_coloring.values():
                if value == 1:
                    mult_color = True
                    break

            for not_contained_action_name, all_its_edges in edge_of_not_choosen_sep.items():
                
                already_seen = False

                if prev_precs is not None:
                    if not_contained_action_name in prev_precs:
                        already_seen = True
                        if prev_precs[not_contained_action_name] == 3:
                            possible_preconditions_for_split[not_contained_action_name] = 3
                            continue

                
                if len(all_its_edges) == 0:
                    if already_seen:
                        possible_preconditions_for_split[not_contained_action_name] = prev_precs[not_contained_action_name]
                    else:
                        possible_preconditions_for_split[not_contained_action_name] = -1
                    continue

                not_contained_color = node_coloring[list(all_its_edges)[0][0]]
                prec_possible = True

                for (u,_) in all_its_edges:

                    if not_contained_color != node_coloring[u]:

                        prec_possible = False
                        break
                
                
                if prev_precs is None:
                    possible_preconditions_for_split[not_contained_action_name] = not_contained_color
                
                if not prec_possible:
                    print(find_key(action_mapping, not_contained_action_name), 'xxx')
                    possible_preconditions_for_split[not_contained_action_name] = 3
                else:
                    if already_seen:
                        if prev_precs[not_contained_action_name] == -1:
                            possible_preconditions_for_split[not_contained_action_name] = not_contained_color
                        elif prev_precs[not_contained_action_name] == not_contained_color:
                            possible_preconditions_for_split[not_contained_action_name] = not_contained_color
                        else:
                            if mult_color:
                                possible_preconditions_for_split[not_contained_action_name] = 3
                            else:
                                possible_preconditions_for_split[not_contained_action_name] = prev_precs[not_contained_action_name]
                            

                    else:
                        possible_preconditions_for_split[not_contained_action_name] = not_contained_color

            for act in action_mapping.values():
                if act not in possible_preconditions_for_split:
                    if act not in zeronary_split[0] and act not in zeronary_split[1]:
                        if prev_precs is not None:
                            if act in prev_precs:
                                    possible_preconditions_for_split[act] = prev_precs[act]
                            else:
                                possible_preconditions_for_split[act] = -1
                        else:
                            possible_preconditions_for_split[act] = -1

            if need_complement:
                for _act, _val in list(possible_preconditions_for_split.items()):
                    if 0 <= _val <= 1:
                        possible_preconditions_for_split[_act] = 1-_val

            

            if zeronay in possible_preconditions:
                possible_preconditions[zeronay].append([zeronary_split, possible_preconditions_for_split])
            else:
                possible_preconditions[zeronay] = [[zeronary_split, possible_preconditions_for_split]]

    print(num_candidates, num_splits)    

    output = {}
    for zeronary in possible_zeronary:
        # TODO key error fix is done here
        output[zeronary] = [zeronary_features[zeronary], possible_preconditions[zeronary]]


    return output, benchmark_val


'''
def zeronary_verification(G_in, action_mapping, prev_features, early_termination):

    # get graph for zeronary features
    G_zero, _ = get_nx_graph_only_action_names_new(G_in)

    all_action_names = [_act for _, _act in action_mapping.items()]

    feature_candidates = prev_features.keys()

    if len(feature_candidates) == 0:
        return True

    all_colorings = dict()

    for _cur_sep in feature_candidates:

        coloring = get_zeronary_coloring(G_zero, set(_cur_sep), all_action_names)

        cs = ColorSplit(set(_cur_sep))
        
        adding_possible = True

        if not coloring is None:
            adding_possible = cs.add(coloring[2][0], coloring[2][1])

        all_splits = cs.get_possible_splits()

        if coloring is None or not adding_possible:
            print('This zeronary seperator is not possible: {} !'.format(_cur_sep))
            # This needs to be changed
            if early_termination:
                return False

        # get preconditions
        new_possible_precs = {}
        for not_contained_action_name, all_its_edges in coloring[1].items():
            
            
            if len(all_its_edges) == 0:
                new_possible_precs[not_contained_action_name] = -1
                continue

            not_contained_color = coloring[0][list(all_its_edges)[0][0]]
            prec_possible = True

            for (u,_) in all_its_edges:
                if not_contained_color != coloring[0][u]:
                    prec_possible = False
                    break
            
            if prec_possible:
                new_possible_precs[not_contained_action_name] = not_contained_color
            else:
                new_possible_precs[not_contained_action_name] = 3


        for prev_split, prev_split_precs in prev_features[_cur_sep][1]:
            complement = None
            for split in all_splits:
                if prev_split == split:
                    complement = False
                    break
                elif prev_split == [split[1], split[0]]:
                    complement = True
                    break

            if complement is None:
                print('For seperator {} is the split {} not contained in a verification instance!'.format(_cur_sep, prev_split))
                if early_termination:
                    return False

            if complement:
                for prec in prev_split_precs:
                    if prec not in new_possible_precs:
                        continue
                    pre_prec, cur_prec = prev_split_precs[prec], new_possible_precs[prec]
                    
                    if pre_prec == -1 and cur_prec > -1:
                        print('For split {} is has a verification instance precondition {}, the learned precondition is {}!'.format(split, cur_prec, pre_prec))
                        if early_termination:
                            return False
                    elif cur_prec == -1 or pre_prec == 3:
                        continue
                    elif pre_prec + cur_prec != 1:
                        print('For split {} is has a verification instance precondition {}, the learned precondition is {}!'.format(split, cur_prec, pre_prec))
                        if early_termination:
                            return False
            else:
                 for prec in prev_split_precs:
                    if prec not in new_possible_precs:
                        continue
                    pre_prec, cur_prec = prev_split_precs[prec], new_possible_precs[prec]
                    
                    if pre_prec == -1 and cur_prec > -1:
                        print('For split {} is has a verification instance precondition {}, the learned precondition is {}!'.format(split, cur_prec, pre_prec))
                        if early_termination:
                            return False
                    elif cur_prec == -1 or pre_prec == 3:
                        continue
                    elif pre_prec != cur_prec:
                        print('For split {} is has a verification instance precondition {}, the learned precondition is {}!'.format(split, cur_prec, pre_prec))
                        if early_termination:
                            return False       
    
    return True
'''


def zeronary_verification(G_in, action_mapping, prev_features, early_termination):

    # get graph for zeronary features
    G_zero, _ = get_nx_graph_only_action_names_new(G_in)

    all_action_names = [_act for _, _act in action_mapping.items()]

    feature_candidates = prev_features.keys()

    if len(feature_candidates) == 0:
        return True

    all_colorings = dict()

    preconditions_possible = True

    for _cur_sep in feature_candidates:

        coloring = get_zeronary_coloring(G_zero, set(_cur_sep), all_action_names)

        cs = ColorSplit(set(_cur_sep))
        
        adding_possible = True

        if not coloring is None:
            adding_possible = cs.add(coloring[2][0], coloring[2][1])

        all_splits = cs.get_possible_splits()

        if coloring is None or not adding_possible:
            print('This zeronary seperator is not possible: {} !'.format(_cur_sep))
            # This needs to be changed
            if early_termination:
                return False

        # get preconditions
        new_possible_precs = {}
        for not_contained_action_name, all_its_edges in coloring[1].items():
            
            
            if len(all_its_edges) == 0:
                new_possible_precs[not_contained_action_name] = -1
                continue

            not_contained_color = coloring[0][list(all_its_edges)[0][0]]
            prec_possible = True

            for (u,_) in all_its_edges:
                if not_contained_color != coloring[0][u]:
                    prec_possible = False
                    break
            
            if prec_possible:
                new_possible_precs[not_contained_action_name] = not_contained_color
            else:
                new_possible_precs[not_contained_action_name] = 3

        split = coloring[2]
        if split[0] == set() and split[1] == set():
            continue


        for prev_split, prev_split_precs in prev_features[_cur_sep][1]:
            complement = None
            #for split in all_splits:
                #if prev_split == split:
                #    complement = False
                #    break
                #elif prev_split == [split[1], split[0]]:
                #    complement = True
                #    break
            if split[0].issubset(prev_split[0]) and split[1].issubset(prev_split[1]):
                complement = False
            elif split[0].issubset(prev_split[1]) and split[1].issubset(prev_split[0]):
                complement = True


            if complement is None:
                complement[0]
                print('For seperator {} is the split {} not contained in a verification instance!'.format(_cur_sep, prev_split))
                if early_termination:
                    return False

            if complement:
                for prec in prev_split_precs:
                    if prec not in new_possible_precs:
                        continue
                    pre_prec, cur_prec = prev_split_precs[prec], new_possible_precs[prec]
                    
                    if pre_prec == -1 and cur_prec > -1:
                        print('For split {} is has a verification instance precondition {}, the learned precondition is {}!'.format(split, cur_prec, pre_prec))
                        preconditions_possible = False
                        if early_termination:
                            return False
                    elif cur_prec == -1 or pre_prec == 3:
                        continue
                    elif pre_prec + cur_prec != 1:
                        print('For split {} is has a verification instance precondition {}, the learned precondition is {}!'.format(split, cur_prec, pre_prec))
                        preconditions_possible = False
                        if early_termination:
                            return False
            else:
                 for prec in prev_split_precs:
                    if prec not in new_possible_precs:
                        continue
                    pre_prec, cur_prec = prev_split_precs[prec], new_possible_precs[prec]
                    
                    if pre_prec == -1 and cur_prec > -1:
                        print('For split {} is has a verification instance precondition {}, the learned precondition is {}!'.format(split, cur_prec, pre_prec))
                        preconditions_possible = False
                        if early_termination:
                            return False
                    elif cur_prec == -1 or pre_prec == 3:
                        continue
                    elif pre_prec != cur_prec:
                        print('For split {} is has a verification instance precondition {}, the learned precondition is {}!'.format(split, cur_prec, pre_prec))
                        preconditions_possible = False
                        if early_termination:
                            return False       
    
    return preconditions_possible


# BFS like coloring 
def get_zeronary_coloring(G, contained_seps:set, all_separators):

    not_used_seps = set(all_separators)-set(contained_seps)

    # edges of not choosen separators
    edge_of_not_choosen_sep = {x: set() for x in not_used_seps}

    # color of a node 
    node_color = {i: None for i in G.nodes()}

    # color in which a separator starts
    sep_colors = {_sep: None for _sep in contained_seps}

    # use a random node as starting point and set its color
    initial_node = list(G.nodes())[0]
    node_color[initial_node] = 0

    # list that contains node where the color got determined but the node was not visited
    set_color = [initial_node]

    # all nodes that were already visited
    alredy_visited = set()

    # boolean that states whether it is a valid coloring
    valid = True

    seen = set()
                        
    # TODO can probably done better currently see each edge twice
    while len(set_color) > 0 and valid:

        # get node with color that was not visited before
        node = set_color.pop(0)
        
        # get all outgoing edges for the node
        for edge in G.out_edges([node],data='action'):

            # only check if edge not already seen
            if (edge[0], edge[1]) not in seen:

                seen.add((edge[0], edge[1]))

                # get all separators that fit the edge
                seps_for_grounding = contained_seps.intersection(edge[2])        

                # if there is some the color has to switch
                if len(seps_for_grounding) > 0:
                    # set color if not already
                    if node_color[edge[1]] is None:
                        # set node color
                        node_color[edge[1]] = 1 - node_color[edge[0]]
                    # check if correctly colored
                    elif node_color[edge[1]] + node_color[edge[0]] != 1:
                        # no valid separator
                        valid = False
                        #print('HERE IT BREAKS 1')
                        return None
                    
                    # check if all separators start in the correct color
                    for _seps in seps_for_grounding:
                        # set color if not set
                        if sep_colors[_seps] is None:
                            sep_colors[_seps] = node_color[edge[0]]
                        # break if separator starts in wrong color
                        elif sep_colors[_seps] != node_color[edge[0]]:
                            # no valid separator 
                            valid = False
                            #print('HERE IT BREAKS 2')
                            return None
                # if no separator fits the edge nodes need to be colored the same
                else:
                    # color not if not already
                    if node_color[edge[1]] is None:
                        # set node color
                        node_color[edge[1]] = node_color[edge[0]]
                    # check coloring, break if not correct
                    elif node_color[edge[1]] != node_color[edge[0]]:
                        # no valid separator
                        valid = False
                        #print('HERE IT BREAKS 3')
                        return None
                    
                fitting_not_used_seps = not_used_seps.intersection(edge[2])

                if len(fitting_not_used_seps) > 0 and len(seps_for_grounding) > 0:
                   # print('Here it breaks 8')
                    return None
                
                for _not_used_sep in fitting_not_used_seps:
                    edge_of_not_choosen_sep[_not_used_sep].add((edge[0],edge[1]))
                
                if edge[1] not in alredy_visited:
                    if edge[1] not in set_color: 
                        set_color.append(edge[1])
        
        # check all ingoing edges
        for edge in G.in_edges([node],data='action'):

            # only check nodes with larger index 
            if (edge[0], edge[1]) not in seen:

                seen.add((edge[0], edge[1]))

                # get all separators that fit the edge
                #seps_for_grounding = separatores_fit_grounded_action(grounding, contained_seps, edge[2], action_mapping, object_mapping)
                seps_for_grounding = contained_seps.intersection(edge[2])
                
                # if there is some the color has to switch
                if len(seps_for_grounding) > 0:
                    # set color if not already
                    if node_color[edge[0]] is None:
                        # set node color
                        node_color[edge[0]] = 1 - node_color[edge[1]]
                    # check if correctly colored
                    elif node_color[edge[1]] + node_color[edge[0]] != 1:
                        # no valid separator
                        valid = False
                       # print('HERE IT BREAKS 4')
                        # set_color = set()
                        return None
                    # check if all separators start in the correct color
                    for _seps in seps_for_grounding:
                        # set color if not set
                        if sep_colors[_seps] is None:
                            sep_colors[_seps] = node_color[edge[0]]
                        # break if separator starts in wrong color
                        elif sep_colors[_seps] != node_color[edge[0]]:
                            # no valid separator
                            valid = False
                          #  print('HERE IT BREAKS 5')
                            return None
                else:
                    # color not if not already
                    if node_color[edge[0]] is None:
                        # set node color
                        node_color[edge[0]] = node_color[edge[1]]
                    # check coloring, break if not correct
                    elif node_color[edge[0]] != node_color[edge[1]]:
                        # no valid separator
                        #print('HERE IT BREAKS 6')
                        set_color = set()
                        valid = False
                        return None
                    
                fitting_not_used_seps = not_used_seps.intersection(edge[2])

                if len(fitting_not_used_seps) > 0 and len(seps_for_grounding) > 0:
                   # print('Here it breaks 7')
                    return None

                for _not_used_sep in fitting_not_used_seps:
                    edge_of_not_choosen_sep[_not_used_sep].add((edge[0],edge[1]))

                # add node to queue if not already contained or already visited
                if edge[0] not in alredy_visited:
                    if edge[0] not in set_color: 
                        set_color.append(edge[0])
        
        # add current node to visited
        alredy_visited.add(node)

    if valid:
        # TODO this is not good 
        separator_colors = [set(), set()]
        for _sep, _sep_col in sep_colors.items():
            if _sep_col is not None:
                separator_colors[_sep_col].add(_sep)
        return node_color, edge_of_not_choosen_sep, separator_colors
    else:
        return None



    
def preprocessing(domain, all_instances, prev_object, action_mapping, prev_locm_types, prev_zeronary_feat, prev_zeronary_prec, mode, number_states, number_inputs, split_dict):
    
    # create domain paths 
    domain_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), domain)
    
    instance_list, action_mapping, action_arity, benchmark_val = [], None, None, None

    num_examples, size_example = 0,0

    for instance in all_instances:

        # create problem path
        problem_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), instance)

        # create state space and parser
        pddl_holder = mimir_holder(domain_path, problem_path)

        # create clingo input
        # TODO the fuction 'create_clingo_input' needs to be replace by a new fuction that uses *all_actions* as input

        

        # get state space as nx graph, edges are labeled with 'action' where this is the action that corresponds to the transition
        graphs_and_actions = []
        all_actions_in_all_traces = set()
        for num_input in range(number_inputs):
            if mode == 'fg':
                G, all_actions = get_nx_graph_from_state_space(pddl_holder, False)
                print(G)
            elif mode == 'pg':
                G, all_actions, _ = bfs_state_space(pddl_holder, number_states, num_input, False)
                print(G)
            elif mode == 'rl':
                G, all_actions, _ = get_trace_rl(pddl_holder, number_states, num_input, False)
                print(G)
            elif mode == 'st':
                G, all_actions, _ = get_trace_simple(pddl_holder, number_states, num_input, False)
                print(G)
            else:
                return None
            
            for _act in all_actions:
                all_actions_in_all_traces.add(_act)
                #all_actions_in_all_traces = all_actions_in_all_traces | all_actions
            graphs_and_actions.append([G,all_actions])
            
            num_examples += 1
            size_example += len(G.edges())

            if mode == 'fg':
                break
    
        # get all objects of this domain

        object_mapping = pddl_holder.get_object_mapping()

        # get all actions of this domain

        action_mapping, action_arity = pddl_holder.get_action_mapping_and_arity()

        # locm_types, object_types = get_locm_like_types_new(all_actions_in_all_traces, action_arity, action_mapping, object_mapping, prev_locm_types)

        prev_locm_types = learn_locm_type(all_actions_in_all_traces, action_arity, object_mapping, prev_locm_types)

        print(action_mapping)
        print(object_mapping)
        print(prev_locm_types)
        print(all_actions_in_all_traces)

        instance_list.append([object_mapping, graphs_and_actions, all_actions_in_all_traces])

    # print(locm_types)
    locm_types = prev_locm_types

    for object_map, instance_graphs_and_actions, all_acts in instance_list:

        object_types = get_locm_types(all_acts, object_map, locm_types)

        for G, all_actions_in_trace in instance_graphs_and_actions:
            
            print('How often are we here')
            
            # get dictionary with: key = mapping value of action, value = arity of the action
            pattern_groundings = get_pattern_groundings(action_arity, all_actions_in_trace)

            zeronary_features, benchmark_val = get_zeronary_features(G, action_mapping, prev_zeronary_feat, benchmark_val)
            zero_passed_pattern = benchmark_val.get_passed_pattern()

            # get all states
            #states_objects = pddl_problem.get_states()
            #states = [pddl_problem.get_state_index(state) for state in states_objects]

            # action_dict_test = {trans.get_creating_action() for state in states for trans in pddl_problem.get_forward_transitions(state)}

            # get set of all actions 
            # action_list = {str(trans.get_creating_action()) for state in states for trans in pddl_problem.get_forward_transitions(state)}

            action_list_test = count_action_groundings_new(G, all_actions_in_trace)

            time_start = time.time()
            #if args.processes is None:
            #    all_atoms, output_obj = get_seperators_and_graphs(G,locm_types, object_types, action_arity, action_list_test, action_mapping, object_mapping, pattern_groundings)
            #else:
            all_atoms, output_obj, split_dict, benchmark_val = get_seperators_and_graphs_parallel(G,locm_types, object_types, action_arity, action_list_test, action_mapping, 
                                                                                    object_map, pattern_groundings, args.processes, prev_object, split_dict, benchmark_val)
            benchmark_val.sum_passed_pattern(zero_passed_pattern)
            time_end = time.time()

            #prev_locm_types = locm_types
            prev_object = output_obj
            prev_zeronary_feat = zeronary_features

    output = ''
    
    zeronary_feature_num = 1
    for zeronary, split_list in zeronary_features.items():
        benchmark_val.add_final_feature_num()
        for [[_add, _delete], zero_prec] in split_list[1]:
            zero_add = [find_key(action_mapping, _act) for _act in _add]
            zero_delete = [find_key(action_mapping, _act) for _act in _delete]
            zero_neg_prec = [find_key(action_mapping, _act) for _act, _val in zero_prec.items() if _val == 0]
            zero_pos_prec = [find_key(action_mapping, _act) for _act, _val in zero_prec.items() if _val == 1]
            zero_undefined_precs = [find_key(action_mapping, _act) for _act, _val in zero_prec.items() if _val == -1]

            print('\n~~~Zeronary-Feature {}\n~~~~~~~~~~~~~'.format(zeronary_feature_num))
            print('Adds:', str(zero_add))
            print('Dels:', str(zero_delete))
            print('PosPrecs:', str(zero_pos_prec))
            print('NegPrecs:', str(zero_neg_prec))
            print('UndefinedPrecs:', str(zero_undefined_precs))

            output += '~~~Zeronary_Feature {}~~~\nAdds: {}\nDels: {}\nPosPrecs: {}\nNegPrecs: {}\nUndefinedPrecs:{}\n\n'.format(zeronary_feature_num, zero_add, zero_delete, 
                                                                                                                                str(zero_pos_prec), str(zero_neg_prec),str(zero_undefined_precs))

            zeronary_feature_num += 1
    
    seps_that_are_undefined = dict()
    
    for atom_num, [_sep, _, _] in all_atoms.items():
        if _sep in seps_that_are_undefined:
            seps_that_are_undefined[_sep].add(atom_num)
        else:
            seps_that_are_undefined[_sep] = {atom_num}

    for _sep, positions in seps_that_are_undefined.items():
        if len(positions) > 1:
            for _pos in positions:
                del all_atoms[_pos]
    
    atom_index = 1

    for atom_num, [_, split, preconditions] in all_atoms.items():
        
        benchmark_val.add_final_feature_num()

        add_list = [(find_key(action_mapping, _add[0]), _add[1]) for _add in split[0]]
        del_list = [(find_key(action_mapping, _del[0]), _del[1]) for _del in split[1]]
        pos_precs, neg_precs, undef_precs = [], [], []
        
        for prec, prec_val in preconditions.items():

            if prec_val == 1:
                pos_precs.append((find_key(action_mapping, prec[0]), prec[1]))
            elif prec_val == 0:
                neg_precs.append((find_key(action_mapping, prec[0]), prec[1]))
            elif prec_val == -1:
                undef_precs.append((find_key(action_mapping, prec[0]), prec[1]))

        print('\n~~~Feature {}\n~~~~~~~~~~~~~'.format(atom_index))
        print('Adds:', str(add_list))
        print('Dels:', str(del_list))
        print('PosPrecs:', str(pos_precs))
        print('NegPrecs:', str(neg_precs))
        print('UndefinedPrecs:', str(undef_precs))
        output += '~~~Feature {}~~~\nAdds: {}\nDels: {}\nPosPrecs: {}\nNegPrecs: {}\nUndefinedPrecs: {} \n\n'.format(atom_index, add_list, del_list, pos_precs, neg_precs, undef_precs)

        atom_index += 1 

    print('Overall Time:', time_end - time_start)

    return output_obj, output, action_mapping, locm_types, prev_zeronary_feat, prev_zeronary_prec, size_example//num_examples, benchmark_val


def verify_solution(domain, instance, action_mapping, output_obj, locm_type, zeronary_features, zeronary_preconditions, size, number_inputs, mode, early_termination):

    print('Verification of {}!'.format(instance))

    domain_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), domain)

    # instance paths 
    verification_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), instance)

    # create state space and parset
    verification_pddl_holder = mimir_holder(domain_path, verification_path)

    # create clingo input
    #action_mapping_verify, object_mapping_verify, _, _ = create_clingo_input(verification_problem, verification_parser)

    # get graph with only action names for zeronary verification
    # get state space as nx graph, edges are labeled with 'action' where this is the action that corresponds to the transition
    graphs_and_actions = []
    all_actions_in_all_traces = set()
    for num_input in range(number_inputs):
        if mode == 'fg':
            G, all_actions = get_nx_graph_from_state_space(verification_pddl_holder, False)
            print(G)
        elif mode == 'pg':
            G, all_actions, _ = bfs_state_space(verification_pddl_holder, size, num_input, False)
            print(G)
        elif mode == 'rl':
            G, all_actions, _ = get_trace_rl(verification_pddl_holder, size, num_input, False)
            print(G)
        elif mode == 'st':
            G, all_actions, _ = get_trace_simple(verification_pddl_holder, size, num_input, False)
            print(G)
        elif mode == 'nfg':
            G, all_actions = get_nx_graph_from_state_space(verification_pddl_holder, True)
            print(G)
        elif mode == 'npg':
            G, all_actions, _ = bfs_state_space(verification_pddl_holder, size, num_input, True)
            print(G)
        elif mode == 'nrl':
            G, all_actions, _ = get_trace_rl(verification_pddl_holder, size, num_input, True)
            print(G)
        elif mode == 'nst':
            G, all_actions, _ = get_trace_simple(verification_pddl_holder, size, num_input, True)
            print(G)
        else:
            return None
        all_actions_in_all_traces = all_actions_in_all_traces | all_actions
        graphs_and_actions.append([G,all_actions])
        
        if mode == 'fg' or mode == 'nfg':
            break

    # G_zero_verify = get_nx_graph_only_action_names(verification_problem, action_mapping)
    
    # get all objects of this domain

    object_mapping_verify = verification_pddl_holder.get_object_mapping()

    # get all actions of this domain

    action_mapping, action_arity_verify = verification_pddl_holder.get_action_mapping_and_arity()

    #locm_types_verify, object_types_verify = get_locm_like_types_new(all_actions_in_all_traces, action_arity_verify, action_mapping, object_mapping_verify, locm_type)

    locm_return = verify_locm_types(all_actions_in_all_traces, object_mapping_verify, locm_type)
    
    if locm_return is None:
        print('Types of this Domain are more restrictive than the learned types!\nStopping verification of this instance!')
        return False

    object_types_verify, object_mapping_verify = locm_return

    # TODO I dont know if this can happen
    if object_types_verify is None:
        print('Types of this Domain are more restrictive than the learned types!\nStopping verification of this instance!')
        return None

    negative_modes = ['nst', 'nrl', 'nfg', 'npg']
    # do zeronary verification
    zeronary_verification_val = True
    # zeronary_verification_val = zeronary_verification(verification_problem, zeronary_features, zeronary_preconditions, action_mapping) 

    for G_verify, all_actions_verify in graphs_and_actions:
        
        # pattern_groundings = get_pattern_groundings(action_arity_verify, all_actions)

        zeronary_verification_val = zeronary_verification(G_verify, action_mapping, zeronary_features, early_termination)

        #if not zeronary_verification_val and mode not in negative_modes:
        #    return zeronary_verification_val
        if not zeronary_verification_val:
            if mode not in negative_modes:
                return False
            else: 
                verify_val = True
                continue


        pattern_groundings_verify = get_pattern_groundings(action_arity_verify, all_actions_verify)

        # get all states
        #states_objects_verify = verification_problem.get_states()
        #states_verify = [verification_problem.get_state_index(state) for state in states_objects_verify]

        # get set of all actions 
        #action_list_verify = {str(trans.get_creating_action()) for state in states_verify for trans in verification_problem.get_forward_transitions(state)}

        #action_dict_test_verify = {trans.get_creating_action() for state in states_verify for trans in verification_problem.get_forward_transitions(state)}

        action_list_test_verify = count_action_groundings_new(G_verify, all_actions_verify)

        # locm_types_verify, object_types_verify = get_locm_like_types(action_dict_test_verify, action_arity_verify, action_mapping, object_mapping_verify, locm_type)

        # action_list_test_verify = count_action_groundings(verification_problem, states_verify, action_mapping, object_mapping_verify)
        if mode in negative_modes:
            verify_value = False
            print('WeGetHere')
        else:
            verify_value = True

        verify_val = verify_parallel(output_obj, G_verify, locm_type, object_types_verify, action_arity_verify, action_list_test_verify, 
                                     action_mapping, object_mapping_verify, pattern_groundings_verify, args.processes, early_termination,
                                     verify_value)

        print(verify_val)

        if verify_val:
            print('VERIFICATION WAS SUCCESSFULL')
        else:
            print('THE VERIFICATION WAS NOT SUCCESSFULL!!!')
            return verify_val


    return verify_val


def gen_clingo_mapping(max_inputsize, max_feature_size):
    mapping = ''

    right_side = ','.join(['O{}'.format(i) for i in range(max_inputsize)])
    left_side = 'mapping((' + right_side + '),'
    for feature_size in range(1,max_feature_size+1):
        for pattern in itertools.product(range(max_inputsize), repeat=feature_size):
            cur_pattern = ','.join([str(index) for index in pattern])
            cur_objtuple = ','.join(['O' + str(index) for index in pattern])
            mapping += left_side + '({}),({})) :- pos_action(_,({})), O{} > -1.\n'.format(cur_pattern, cur_objtuple, right_side, max(list(pattern)))
            mapping += left_side + '({}),({})) :- neg_action(_,({})), O{} > -1.\n'.format(cur_pattern, cur_objtuple, right_side, max(list(pattern)))

    mapping_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mapping.lp")
    
    with open(mapping_path, "w") as text_file:
        text_file.write(mapping)

    return mapping_path
    #mapping((O0,O1,O2),(2),O2) :-     pos_action(_,(O0,O1,O2)), O2 > -1.
    #mapping((O0,O1,O2),(2,2),(O2,O2)) :-pos_action(_,(O0,O1,O2)), O2 > -1.
    #mapping((O0,O1,O2),(2),O2) :-     neg_action(_,(O0,O1,O2)), O2 > -1.
    #mapping((O0,O1,O2),(0,0),(O0,O0)) :-neg_action(_,(O0,O1,O2)), O0 > -1.







if __name__ == '__main__':
    
    # get domain and instance 
    args = get_arguments()
    
    output_object, action_mapping, locm_types, zeronary_feat, zeronary_prec, split_dict = None, None, None, None, None, None
    # for instance in args.instance:
    output_object, output, action_mapping, locm_types, zeronary_feat, zeronary_prec, avg_size, bench = preprocessing(args.domain, args.instance, output_object, 
                                                                                                        action_mapping, locm_types, zeronary_feat, 
                                                                                                        zeronary_prec, args.learning_mode, args.learning_size, 
                                                                                                        args.learning_number_inputs, split_dict)
    
    dir_path = os.path.dirname(os.path.realpath(__file__))

    if not os.path.exists(dir_path+"/output/"):
        os.makedirs(dir_path+"/output/")

    if args.output is None:
        out_path = dir_path + '/output/' + 'output.txt'
    else:
        out_path = dir_path + '/output/' + args.output + '.txt'
    
    with open(out_path, "w") as text_file:
        text_file.write(output)

    verification_val = True

    if args.verification_instance is not None:
        verification_cases = get_verification_instances(args.verification_instance)

        print('Length of Verifivation cases:',len(verification_cases))

        for v_case_num, v_case in enumerate(verification_cases):
            print('VERIFY NUMBER {}'.format(v_case_num))
            verification_val = verify_solution(args.domain, v_case.get_instance(), action_mapping, output_object, locm_types, zeronary_feat, zeronary_prec,
                                                v_case.get_input_size(), v_case.get_number_traces(), v_case.get_mode(), v_case.get_early_termination()) and verification_val

        #verification_val = True
        #if args.verification_instance is not None:
        #    for v_instance in args.verification_instance:
        #        verification_val = verify_solution(args.domain, v_instance, action_mapping, output_object, locm_types, zeronary_feat, zeronary_prec,
        #                                           args.verification_size, args.verification_number_inputs, args.verification_mode, args.verification_termination) and verification_val
    
    print(verification_val)
    print('??????????{},{},{}??????????'.format(avg_size, bench.get_pattern_combi_length(), bench.get_passed_pattern()))

    if verification_val:
        sys.exit(0)
    else:
        sys.exit(1)

    '''
    # create domain and instance paths 
    domain_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.domain)
    problem_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.instance)

    # create state space and parset
    pddl_problem = StateSpace.create(domain_path,problem_path)
    pddl_parser = PDDLParser(domain_path, problem_path)

    # create clingo input
    action_mapping, object_mapping, clingo_input_path, pattern_dictionary = create_clingo_input(pddl_problem, pddl_parser)
    
    # get dictionary with: key = mapping value of action, value = arity of the action
    action_arity = {action_mapping[_act.get_name()]: _act.get_arity() for _act in pddl_parser.get_domain().get_actions()}
    
    zeronary_features, zeronary_preconditions = get_zeronary_features(problem=pddl_problem, action_mapping=action_mapping)

    # get state space as nx graph, edges are labeled with 'action' where this is the action that corresponds to the transition
    G, all_actions = get_nx_graph_from_state_space(pddl_problem, action_mapping, object_mapping)

    pattern_groundings = get_pattern_groundings(action_arity, all_actions)

    action_dict_test = {trans.get_creating_action() for state in states for trans in pddl_problem.get_forward_transitions(state)}

    locm_types, object_types = get_locm_like_types(action_dict_test, action_arity, action_mapping, object_mapping)


    # get all states
    states_objects = pddl_problem.get_states()
    states = [pddl_problem.get_state_index(state) for state in states_objects]

    # get set of all actions 
    action_list = {str(trans.get_creating_action()) for state in states for trans in pddl_problem.get_forward_transitions(state)}
    
    print('action_arity', action_arity)
    '''

    # load files to clingo 
    '''
    ctl = Control(arguments=['--opt-mode=opt'])

    ctl.load(clingo_input_path)
    ctl.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'statics.lp'))
    ctl.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'optimisation.lp'))

    # ground and solve clingo program
    ctl.ground()
    clingo_result = ctl.solve(yield_=True, on_model=lambda m: print("Answer: {}".format(m.cost)), 
                              on_finish=lambda m: print("Final Solution: {}".format(m.exhausted)))
    
    
    # get all possible solutions 
    #clingo_resultes = clingo_result.get()

    # the last solution is the optimal (TODO need to check this)
    optimal_solution = None
    with clingo_result as res: 
        for m in res:
            optimal_solution = m

    print('\n########## statics ########## \n', optimal_solution, '\n')
    '''
    # calculate zeronary predicates 
    # zeronary_predicates = step_2(pddl_problem, pddl_parser)

    # print('########## zeronary predicates ########## \n', zeronary_predicates, '\n')

    # statics, statics_pattern = get_statics_with_pattern(optimal_solution)

    # if statics is not None:
    #    unary_statics = {_static for _static, _arity in statics.items() if _arity == 1}
    
    '''
    action_dict_test = {trans.get_creating_action() for state in states for trans in pddl_problem.get_forward_transitions(state)}

    locm_types, object_types = get_locm_like_types(action_dict_test, action_arity, action_mapping, object_mapping)

    action_list_test = count_action_groundings(pddl_problem, states, action_mapping, object_mapping)

    time_start = time.time()
    if args.processes is None:
        all_atoms, output_obj = get_seperators_and_graphs(G,locm_types, object_types, action_arity, action_list_test, action_mapping, object_mapping, pattern_groundings)
    else:
        all_atoms, output_obj = get_seperators_and_graphs_parallel(G,locm_types, object_types, action_arity, action_list_test, action_mapping, object_mapping, pattern_groundings, args.processes, None)

    time_end = time.time()

    output = ''

    
    for feature_num, (zeronary, [_add, _delete]) in enumerate(zeronary_features.items()):

        zero_add = [find_key(action_mapping, _act) for _act in _add]
        zero_delete = [find_key(action_mapping, _act) for _act in _delete]
        zero_pos_prec = [find_key(action_mapping, _act) for _act in zeronary_preconditions[zeronary][0]]
        zero_neg_prec = [find_key(action_mapping, _act) for _act in zeronary_preconditions[zeronary][1]]


        print('\n~~~Zeronary-Feature {}\n~~~~~~~~~~~~~'.format(feature_num))
        print('Adds:', str(zero_add))
        print('Dels:', str(zero_delete))
        print('PosPrecs:', str(zero_pos_prec))
        print('NegPrecs:', str(zero_neg_prec))

        output += '~~~Zeronary_Feature {}~~~\nAdds: {}\nDels: {}\nPosPrecs: {}\nNegPrecs: {}\n\n'.format(feature_num, zero_add, zero_delete, zero_pos_prec, zero_neg_prec)

    for atom_num, [_, split, preconditions] in all_atoms.items():

        add_list = [(find_key(action_mapping, _add[0]), _add[1]) for _add in split[0]]

        del_list = [(find_key(action_mapping, _del[0]), _del[1]) for _del in split[1]]

        pos_precs, neg_precs = [], []
        
        for prec, prec_val in preconditions.items():

            if prec_val == 1:
                pos_precs.append((find_key(action_mapping, prec[0]), prec[1]))
            elif prec_val == 0:
                neg_precs.append((find_key(action_mapping, prec[0]), prec[1]))

        print('\n~~~Feature {}\n~~~~~~~~~~~~~'.format(atom_num))
        print('Adds:', str(add_list))
        print('Dels:', str(del_list))
        print('PosPrecs:', str(pos_precs))
        print('NegPrecs:', str(neg_precs))
        output += '~~~Feature {}~~~\nAdds: {}\nDels: {}\nPosPrecs: {}\nNegPrecs: {}\n\n'.format(atom_num, add_list, del_list, pos_precs, neg_precs)

    print('Overall Time:', time_end - time_start)
    
    '''

    '''
    if args.verification_instance is not None:

        domain_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.domain)

        # instance paths 
        verification_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.verification_instance)

        # create state space and parset
        verification_problem = StateSpace.create(domain_path,verification_path)
        verification_parser = PDDLParser(domain_path, verification_path)

        # create clingo input
        action_mapping_verify, object_mapping_verify, _, _ = create_clingo_input(verification_problem, verification_parser)

        # get dictionary with: key = mapping value of action, value = arity of the action
        action_arity_verify = {action_mapping[_act.get_name()]: _act.get_arity() for _act in verification_parser.get_domain().get_actions()}

        # get graph with only action names for zeronary verification
        G_zero_verify = get_nx_graph_only_action_names(verification_problem, action_mapping)

        # do zeronary verification
        zeronary_verification_val = zeronary_verification(verification_problem, zeronary_features, zeronary_preconditions, action_mapping) 

        if not zeronary_verification_val:

            print('Zeronary verification was not successfull!!\nStopping verification...')

        else:
            # get state space as nx graph, edges are labeled with 'action' where this is the action that corresponds to the transition
            G_verify, all_actions_verify = get_nx_graph_from_state_space(verification_problem, action_mapping, object_mapping_verify)

            pattern_groundings_verify = get_pattern_groundings(action_arity_verify, all_actions_verify)

            # get all states
            states_objects_verify = verification_problem.get_states()
            states_verify = [verification_problem.get_state_index(state) for state in states_objects_verify]

            # get set of all actions 
            action_list_verify = {str(trans.get_creating_action()) for state in states_verify for trans in verification_problem.get_forward_transitions(state)}

            action_dict_test_verify = {trans.get_creating_action() for state in states_verify for trans in verification_problem.get_forward_transitions(state)}

            locm_types_verify, object_types_verify = get_locm_like_types(action_dict_test_verify, action_arity_verify, action_mapping, object_mapping_verify)

            action_list_test_verify = count_action_groundings(verification_problem, states_verify, action_mapping, object_mapping_verify)

            verify_val = verify_parallel(output_obj, G_verify, locm_types_verify, object_types_verify, action_arity_verify, action_list_test_verify, action_mapping, object_mapping_verify, pattern_groundings_verify, args.processes)

            if verify_val:
                print('VERIFICATION WAS SUCCESSFULL')
            else:
                print('THE VERIFICATION WAS NOT SUCCESSFULL!!!')
    '''
        


