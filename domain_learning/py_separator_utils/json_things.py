import copy
import itertools
import math
import random
import json
from collections import defaultdict
from warnings import catch_warnings

# todo should have some locm style types, s.t. better learning for negated effects
class JSONThings:

    def __init__(self, trace_file_path, add_statics):

        self.trace_path = trace_file_path
        self.trace = self.parse_trace(trace_file_path, max_traces=1000)
        self.add_statics = add_statics
        
        print("TRACE LENGTH:", len(self.trace))
        if len(self.trace) > 0:
            print("FIRST TRACE ELEMENT KEYS:", self.trace[0].keys())
        
        # get actions
        self.all_actions = set()
        for action_pair in self.trace:
            self.all_actions.add(action_pair['action_name'])

        self.action_dict = dict()
        for pos, action in enumerate(list(self.all_actions)):
            self.action_dict[pos] = action

        # get arity
        self.action_arity = {action: -1 for action in self.action_dict.values()}
        for action_pair in self.trace:
            try:
                if self.action_arity[action_pair['action_name']] == -1:
                    self.action_arity[action_pair['action_name']] = len(action_pair['action_objects'])
                else:
                    if self.action_arity[action_pair['action_name']] != len(action_pair['action_objects']):
                        raise ValueError
            except KeyError:
                for act in self.all_actions:
                    self.action_arity[act] = 0
                break
        print("%%% Actions \n", self.all_actions, '\n', self.action_dict, '\n', self.action_arity, '\n')

        ### get objects
        self.all_objects = set()
        for action_pair in self.trace:
            for relation in action_pair['state']:
                for grounding in action_pair['state'][relation]:
                    self.all_objects.update(set(grounding))
                #if relation == 'detected_objects':
                #    for grounding in action_pair['state'][relation]:
                #        self.all_objects.add(grounding)
                #elif relation == 'detected_object_names':
                #    continue
                #else:
                #    for grounding in action_pair['state'][relation]:
                #        self.all_objects.update(set(grounding))


        self.object_dict = dict()
        for obj in self.all_objects:
            self.object_dict[obj] = obj

        print("%%% Objects \n", self.all_objects, '\n', self.object_dict, '\n')

        self.statics = set()

        # Assumption that every predicate is contained in the first state
        self.feature_arity_dict = dict()
        for j in range(len(self.trace)):
            for predicate in self.trace[j]['state']:
                if predicate == 'detected_objects' or predicate == 'detected_object_names' : continue
                # todo remove again
                if len(self.trace[j]['state'][predicate]) == 0: continue
                self.feature_arity_dict[predicate] = len(self.trace[j]['state'][predicate][0])
        self.feature_arity = [(feat, ar) for feat, ar in self.feature_arity_dict.items()]

        self.set_empty_predicates()

        print("%%% Feature-Arity \n", self.feature_arity_dict, '\n', self.feature_arity, '\n')

        self.node_dict = dict()
        self.parsed_node_dict = dict()

        # .. only used for verification
        # self.seen_applicable_actions = {act: set() for (act, _) in self.action_arity}
        # self.seen_non_applicable_actions = {act: set() for (act, _) in self.action_arity}
        '''
            We can assume that we know the static predicates
        '''
        # todo there are no static predicates, check whetehr this changes anything
        self.unary_static_predicates = {0}

        '''
            ToDo The following function should be done new
        '''
        # self.predicate_types = self.set_predicate_types()
        # print(self.predicate_types)

        # todo Check this function
        self.objects_types = {0: self.all_objects}
        self.state_action_ground_action = dict()

        self.effect_counter = None
        try:
            self.effects = self.infer_effects()
        except NotImplementedError:
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('The effects are not well formed!!!')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        print('test')
        

    def combine_dicts(self, d1, d2):
        combi_dict = dict()
        for f in d1:
            combi_dict[f] = d1[f]
            if f in d2:
                for p in d2[f]:
                    if f not in combi_dict[f]:
                        combi_dict[f]


        for f in d2:
            if f not in combi_dict:
                combi_dict[f] = d2[f]

        return combi_dict

    def count_predicate(self, counter, things):

        for pred in things:
            if pred not in counter: counter[pred] = dict()
            for x in things[pred]:
                x_tuple = tuple(x)
                if x_tuple in counter[pred]:
                    counter[pred][x_tuple] += 1
                else:
                    counter[pred][x_tuple] = 1

        return counter

    def normalize_state(self, state):
        new_state = {}

        for pred, groundings in state.items():
            if pred in ["detected_objects", "detected_object_names"]:
                new_state[pred] = groundings
                continue

            cleaned = []
            for grounding in groundings:
                unique = list(dict.fromkeys(grounding))  
                cleaned.append(unique)

            new_state[pred] = cleaned

        return new_state

    # todo make this such that we give the file name
    def parse_trace(self, trace_file_path, max_traces=None):
        with open(trace_file_path, 'r') as file:
            trace = json.load(file)
        if max_traces is not None:
            trace = trace[:max_traces]
        num_none = 0
        trace_without_none = []

        new_state = dict()
        #for i in range(len(trace)):
        #    trace[i]['state'] = self.normalize_state(trace[i]['state'])
        for position in range(len(trace)-1):

            """
            if trace[position+1]['action-name'] == 'none':
                new_state = self.count_predicate(new_state, trace[position]['state'])
                num_none += 1
                continue

            next_state = {pred: [list(x) for x in new_state[pred] if new_state[pred][x] > math.floor(0.9 * num_none)
                                 ]for pred in new_state}

            new_dict = {'action_name': trace[position+1]['action-name'],
                            'action_objects': trace[position+1]['action-objects'],
                            'state': next_state}

            print(trace[position+1]['action-name'], num_none, new_state)
            new_state = self.count_predicate(dict(), trace[position]['state'])
            num_none = 0
            #new_state = trace[position]['state']

            trace_without_none.append(new_dict)
            """

            trace[position]['action_name'] = trace[position+1]['action-name']
            trace[position]['action_objects'] = trace[position+1]['action-objects']
            
            trace[position]['action_objects'].append('objrobot')

        trace = trace[:-1]
        #print('Overall Length', len(trace))
        #print('Not none Length', len(trace_without_none))

        return trace

    def set_empty_predicates(self):
        for s in self.trace:
            for pred in self.feature_arity_dict:
                if pred not in s['state']:
                    s['state'][pred] = []
            s['detected_object_names'] = []
            s['detected_objects'] = []



    def get_object_types(self):
        return self.objects_types

    # todo we can have object types to make it faster, locm this on predicate positions, and argument positions
    # currently all objects have the same type
    def set_object_types(self):
        raise NotImplementedError

    # all predicate positions have the same type
    # todo add this back when the object types are given
    # Todo check whether there needs to be some dummy predicate types
    def set_predicate_types(self):
        raise NotImplementedError

    # todo maybe need to check whether there are static predicates in the input
    # see whether this needs to be implemented
    def get_unary_statics(self):
        raise NotImplementedError

    def get_action_name_by_index(self, index: int):
        return self.action_dict[index]

    def get_object_by_index(self, index: int):
        return self.object_dict[index]

    def get_object_set(self):
        return set(self.object_dict.keys())

    def get_state_space(self):
        raise NotImplementedError

    def get_length(self):
        return len(self.trace)

    def get_effect_counter(self):
        return self.effect_counter

    # todo rename this
    def sample_trace_from_init(self):

        state_trace, action_trace = [], []

        for pos, state_action in enumerate(self.trace):
            action_trace.append(state_action['action_name'])
            state_trace.append(pos)

        return state_trace, action_trace

    # new action object list
    def get_action_object_list(self):
        action_object = []
        for state in self.trace:
            action_object.append(state['action_objects'])
        return action_object

    # only needed for pddl verification
    def sample_bfs_from_init(self, length: int):
        raise NotImplementedError

    # After adding back action, it should be checked what the effects are ???
    def get_effects_of_action(self, action):
        raise NotImplementedError

    # only needed for pddl verification
    def get_random_state(self):
        raise NotImplementedError

    # only needed for pddl verification
    def get_random_state_action_pair(self, positive, action_name):
        raise NotImplementedError

    # only needed for pddl verification
    def get_all_reduced_applicable_actions(self, state_index, positive, cur_action_name):
        raise NotImplementedError

    # only needed for pddl verification
    def get_all_applicable_actions(self, index):
        raise NotImplementedError
    
    # Todo implement this
    # see when this is used
    def parse_states(self):
        raise NotImplementedError

    # ???
    def parse_state_precondition(self, index):
        raise NotImplementedError

    # ???
    def parse_state_precondition_test(self, index):

        state_dict = {pred: set() for pred in self.feature_arity_dict}

        state = self.trace[index]
        for pred in state['state']:
            if pred not in self.feature_arity_dict: continue
            for grounding in state['state'][pred]:
                state_dict[pred].add(tuple(grounding))

        return state_dict

    # todo check whether this is a getter or what
    # need to be implemented for  input
    def action_arities(self):
        raise NotImplementedError

    def get_action_arity(self):
        return self.action_arity

    # todo move the funciton from the constructor here
    def feature_arities(self):
        raise NotImplementedError

    def get_feature_arity(self):
        return self.feature_arity

    # we cant do this, since there are no statics
    # todo need to be done when statics are given
    def initialize_state_with_statics(self, in_dicts):
        raise NotImplementedError

    # todo need to rewrite this for the new input
    def parse_state_with_dicts(self, in_dicts, state_index):
        #print(in_dicts)
        for ar in in_dicts:
            for pred in in_dicts[ar]:

                for grounding in self.trace[state_index]['state'][pred]:

                    for mask in in_dicts[ar][pred]:

                        partial_tuple = tuple([None if mask[obj_num] == 0
                                           else -1 if mask[obj_num] == -1
                                           else obj
                                           for obj_num, obj in enumerate(grounding)])

                        identified_argument = grounding[mask.index(-1)]

                        if partial_tuple in in_dicts[ar][pred][mask]:
                            if identified_argument in in_dicts[ar][pred][mask][partial_tuple]:
                                in_dicts[ar][pred][mask][partial_tuple][identified_argument] += 1
                            else:
                                in_dicts[ar][pred][mask][partial_tuple][identified_argument] = 1
                        else:
                            in_dicts[ar][pred][mask][partial_tuple] = {identified_argument: 1}
        return in_dicts

    # todo see where this is used
    def find_static_unary_preconditions(self):
        raise NotImplementedError

    # todo see where this is used
    def get_preconditions(self):
        raise NotImplementedError

    # todo this should be added back when we use statics
    def set_predicate_types_for_trace(self, state_trace):
        raise NotImplementedError

    # todo check this function
    # this function fails (NotImplementedError) iff there are not well formed effects, e.g.
    # there is the same action name with different number of effects for the same predicate and sign
    def get_effects(self):
        return self.effects

    def clean_state(self, state, next_objs: set):
        new_state = set()
        for pred in state:
            args = set(pred)
            if len(args) == len(next_objs.intersection(args)):
                new_state.add(pred)
        return new_state

    def infer_effects(self):
        effects = []
        number_of_effects = {}
        effect_count = {act: dict() for act in self.action_arity}
        for act_pos, state_action in enumerate(self.trace[:-1]):
            current_effects = dict()
            ### todo remove
            if state_action['action_name'] == '':
                effects.append(None)
                continue
            for feature in self.feature_arity_dict.keys():

                prev_state = {tuple(gr) for gr in state_action['state'][feature]}

                prev_state = self.clean_state(prev_state, set(self.trace[act_pos+1]['state']['detected_objects']))

                next_state = {tuple(gr) for gr in self.trace[act_pos+1]['state'][feature]}

                next_state = self.clean_state(next_state, set(self.trace[act_pos]['state']['detected_objects']))

                current_effects[feature] = {0:prev_state.difference(next_state),
                                            1:next_state.difference(prev_state)}

            effect_set = set()
            for f in current_effects:
                if len(current_effects[f][0]) > 0:
                    effect_set.add((0, state_action['action_name'], f, len(current_effects[f][0])))
                if len(current_effects[f][1]) > 0:
                    effect_set.add((1, state_action['action_name'], f, len(current_effects[f][1])))

            effect_set = frozenset(effect_set)
            if effect_set in effect_count[state_action['action_name']]:
                effect_count[state_action['action_name']][effect_set] += 1
            else:
                effect_count[state_action['action_name']][effect_set] = 1

            if state_action['action_name'] not in number_of_effects:
                numeric_dict = dict()
                for f in current_effects:
                    numeric_dict[f] = {0: len(current_effects[f][0]),
                                    1: len(current_effects[f][1])}
                number_of_effects[state_action['action_name']] = numeric_dict
            else:
                for f in current_effects:
                    number_of_effects[state_action['action_name']][f][0] = max(
                        number_of_effects[state_action['action_name']][f][0],
                        len(current_effects[f][0])
                    )
                    number_of_effects[state_action['action_name']][f][1] = max(
                        number_of_effects[state_action['action_name']][f][1],
                        len(current_effects[f][1])
                    )

            effects.append(current_effects)

        '''
        for act, eff in effect_count.items():
            print('\n%%%%%%%%%%%  {}  %%%%%%%%%%%%%'.format(act))
            overall = None
            for e, c in eff.items():
                if overall is None:
                    overall = set(e)
                else:
                    overall.intersection_update(set(e))
                print(c, ':', e)
            print(overall)
        '''
        self.effect_counter = number_of_effects

        return effects



if __name__ == '__main__':

    test = JSONThings(None)