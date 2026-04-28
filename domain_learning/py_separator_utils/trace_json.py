import copy
import itertools
import time
import pymimir
from operator import contains
from typing import override
from py_separator_utils.json_things import JSONThings
from py_separator_utils.mimir_things import mimir_thing
from pathlib import Path


class CountUp:
    def __init__(self, n):
        self.n = n
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.n:
            self.current += 1
            return self.current-1
        else:
            raise StopIteration

class JSONTrace:

    def __init__(self, json: JSONThings, dropped_predicate):

        self.state_trace, self.action_trace = json.sample_trace_from_init()

        self.json = json

        self.validation = False
        self.length = json.get_length()
        self.current = 0
        self.dropped_preds = dropped_predicate
        #self.required_action_arity = {}


        # get from traces
        self.action_name_list = copy.deepcopy(self.action_trace)
        # is empty currently
        self.action_object_list = json.get_action_object_list()

        # all 0 currently
        self.action_arity = json.get_action_arity()

        self.parsed_state_dict = dict()

        self.predicate_arity = json.get_feature_arity()
        self.predicate_arity_dict = {name: arity for (name,arity) in self.predicate_arity}

        # dictionary to store queries of appended arguments
        self.argument_queries = {action: {ar: None for ar in range(arity)}
                                 for action, arity in self.action_arity.items()}

        # can get the effects
        self.affected_object_list, self.affected_tuple_list = [], []
        # TODO get the effects

        # map action arguments to effected predicates, where no argument prediction is needed
        self.effect_mapping = set()

        # this should work
        self.candidate_dict = self.predicate_patterns()

        # TODO add this back in the case of static predicates
        # self.initialize_dict(self.candidate_dict)

        # check what this does
        # self.grounding_preconditions_pos, self.grounding_preconditions_neg, self.mask_pre_pos, self.mask_pre_neg = self.get_grounding_preconditions()
        self.validation = False


        # check this
        self.predicate_types = {pred:{i:[] for i in range(ar)} for pred, ar in self.predicate_arity_dict.items()}
        # self.set_predicate_types()

        self.grounding_preconditions_pos, self.grounding_preconditions_neg = None, None
        # todo write function that gets effects for an trace

    def __iter__(self):
        # new iterator for new trace definition
        return CountUp(self.length)

    def set_predicate_types(self):
        raise NotImplementedError

    def get_predicate_types(self):
        return self.predicate_types

    def get_effects_of_action(self, action):
        raise NotImplementedError

    def reset_parsed_dict(self):
        self.parsed_state_dict = dict()


    '''
        TODO do we ever need to reset this dict or can we always use the same dict????
        Parsed state should not be reset
    '''
    def parse_state(self, position):
        # need to work with full groundings in the dictionary such that one can count how many full groundings
        # for a partial grounding there are
        if position in self.parsed_state_dict:
            return self.parsed_state_dict[position]

        r_val = self.json.parse_state_with_dicts(copy.deepcopy(self.candidate_dict), position)
        self.parsed_state_dict[position] = r_val

        return r_val

    def initialize_dict(self, dicts):
        raise NotImplementedError

    def get_action_name(self, index):
        return self.action_name_list[index]

    def get_action_objects(self, index):
        return self.action_object_list[index]

    def get_state_of_index(self, index):
        return self.state_trace[index]

    def print_action_arity(self):
        print(self.action_arity)

    def get_effect_mapping(self):
        return self.effect_mapping

    def set_effect_mapping(self, effect_mapping):
        self.effect_mapping = effect_mapping

    def print_query_output(self):
        for action in self.argument_queries:
            if action == '' or action == 'none':continue
            print('%%%  {}({})  %%%'.format(action, ' '.join(f'x{j}' for j in range(self.action_arity[action]))))
            for position, query in self.argument_queries[action].items():
                if query is None:
                    print('\tx{}: was given is the input'.format(position))
                else:
                    print('\tx{}:'.format(position), query)

    def get_queries(self):
        return self.argument_queries

    '''
    we can make this 'simpler' with using the below function is_contained in position
    '''
    def add_arguments(self, argument_dict, action_name, query):

        #print("THE ALGORITHM TRIES TO ADD SOMETHING!!!")


        if len(set(argument_dict.values())) == 1 and not self.validation:
            # print('THESE SHOULD NOT BE ADDED!')
            # todo add this back!
            if not self.json.add_statics:
                return False

        cur_arity = self.action_arity[action_name]
        already_contained = False

        for position in range(cur_arity):
            already_contained = True
            for action_number, new_arg in argument_dict.items():

                if new_arg != self.action_object_list[action_number][position]:

                    already_contained = False
                    break
            if already_contained:
                break

        if not already_contained or self.validation:
            # print("Added Arguments: ", set(argument_dict.values()))
            self.argument_queries[action_name][self.action_arity[action_name]] = query

            for action_number, new_arg in argument_dict.items():
                self.action_object_list[action_number].append(new_arg)

            self.action_arity[action_name] += 1
        if not self.validation:
            if not already_contained:
                print('Here was something added!')
            return not already_contained
        else:
            return True
    '''
    Checks if the arguments of a query/patterns is already contained in some argument positions
    returns all positions it is cotnained in
    '''
    def is_contained_in_positions(self, argument_dict, action_name):
        possible_positions = set()
        for pos in range(self.action_arity[action_name]):
            contained = True
            for action_number, new_arg in argument_dict.items():
                if new_arg != self.action_object_list[action_number][pos]:
                    contained = False
                    break
            if contained:
                possible_positions.add(pos)
        return possible_positions

    # cannot do this since hidden truth not known
    def check_final_args(self):
        raise NotImplementedError
    
    '''
        TODO Check if this is ok ...
    '''
    def get_action_effects(self):
        return self.json.get_effects()


    def get_action_effect_dict(self):

        effect_dict = copy.deepcopy(self.json.get_effect_counter())
        for _act in effect_dict:
            for _pred in effect_dict[_act]:
                for _sign in list(effect_dict[_act][_pred]):
                    if effect_dict[_act][_pred][_sign] == 0:
                        effect_dict[_act][_pred][_sign] = dict()
                        continue
                    effect_dict[_act][_pred][_sign] = dict()
                    for _pos in range(self.predicate_arity_dict[_pred]):
                        effect_dict[_act][_pred][_sign][_pos] = {i for i in range(self.action_arity[_act])}

        return effect_dict

    '''
        There can be some optimization of the output, when there are more possible patterns than effects for an action
        that means, that there will be a minimal set of action patterns that will describe the effects, (if all 
        arguments contained in STRIPS like labels, this set will have the size of the effects ???)
    '''
    def get_effect_argument_positions(self):

        possible_effect_positions = self.get_action_effect_dict()
        action_effects = self.json.get_effects()

        print(possible_effect_positions)

        for position, action in enumerate(self.action_name_list[:-1]):
            # todo remove this
            if action == '' or action == 'none':
                continue
            effects = action_effects[position]
            action_name = self.action_name_list[position]
            arguments = self.action_object_list[position]
            for pred in effects:
                for sign in [0,1]:
                    if len(effects[pred][sign]) == 0: continue
                    for pos in list(possible_effect_positions[action_name][pred][sign]):
                        eff_args = {gr[pos] for gr in effects[pred][sign]}
                        if len(eff_args) == 0: continue
                        for act_pos, act_arg in enumerate(arguments):
                            if act_arg not in eff_args:
                                possible_effect_positions[action_name][pred][sign][pos].discard(act_pos)

        for _act in possible_effect_positions:
            print('%%%%%% {} %%%%%%'.format(_act))
            print(possible_effect_positions[_act])

        pattern_dict = copy.deepcopy(possible_effect_positions)
        for act in possible_effect_positions:
            for pred in possible_effect_positions[act]:
                for sign in possible_effect_positions[act][pred]:
                    possible_argument_positions = [possible_effect_positions[act][pred][sign][i]
                                                   for i in range(len(possible_effect_positions[act][pred][sign]))]
                    pattern_dict[act][pred][sign] = {pat for pat in itertools.product(*possible_argument_positions)}

        for _act in pattern_dict:
            print('%%%%%% {} %%%%%%'.format(_act))
            print(pattern_dict[_act])

        for position, action in enumerate(self.action_name_list[:-1]):
            # todo remove this
            if action == '' or action == 'none':
                continue

            not_all_effects_covered = False
            for predicate in pattern_dict[action]:
                for sign in pattern_dict[action][predicate]:
                    # check whether each pattern has an corresponding effect
                    groundings_from_patterns = set()
                    for pattern in list(pattern_dict[action][predicate][sign]):
                        pattern_grounding = tuple(
                            [self.action_object_list[position][pattern_pos] for pattern_pos in pattern])

                        if pattern_grounding not in action_effects[position][predicate][sign]:
                            pattern_dict[action][predicate][sign].remove(pattern)
                        else:
                            groundings_from_patterns.add(pattern_grounding)
                    # check whether for each effect there is a pattern_grounding
                    if action_effects[position][predicate][sign] != groundings_from_patterns:
                        print('Not Covered',  action_effects[position][predicate][sign], '|||', groundings_from_patterns)
                        print('-- ', predicate, action)
                        not_all_effects_covered = True
             
            if not_all_effects_covered:
                raise OverflowError
                print("For an effect of the action {} there arguments of effects missing".format(
                    action
                ))

                
                # The effect patterns tell us how many arguments the action MUST have.
                #max_size = max(
                #    len(pattern)
                #    for pred in effects[action]
                #    for pattern in effects[action][pred]
                #)

                # Store how many parameters the action must use.
                #self.required_action_arity[action] = max_size

                #continue
                    
            


        ### reorder such that it fits old style
        new_pattern_dict = {0: dict(), 1:dict()}
        for act in pattern_dict:
            for pred in pattern_dict[act]:
                for sign in pattern_dict[act][pred]:
                    if len(pattern_dict[act][pred][sign]) == 0:
                        continue
                    else:
                        if act not in new_pattern_dict[sign]:
                            new_pattern_dict[sign][act] = dict()
                        new_pattern_dict[sign][act][pred] = pattern_dict[act][pred][sign]


        self.effect_mapping = new_pattern_dict
        return new_pattern_dict

    def predicate_patterns(self):

        out_dict = dict()

        for (pred, pred_ar) in self.predicate_arity:

            if pred_ar == 0:
                continue

            if pred_ar not in out_dict:
                out_dict[pred_ar] = {pred:dict()}

                for x in itertools.product([0,1], repeat=pred_ar-1):
                    current_template = list(x)
                    for i in range(pred_ar):
                        current = copy.deepcopy(current_template)
                        current.insert(i, -1)
                        out_dict[pred_ar][pred][tuple(current)] = dict()
            else:
                out_dict[pred_ar][pred] = dict()
                key = list(out_dict[pred_ar].keys())[0]
                for mask in out_dict[pred_ar][key]:
                    out_dict[pred_ar][pred][mask] = dict()

        return out_dict

    # probably validation
    # todo check what this function is dooing
    def set_grounding_preconditions(self):
        self.grounding_preconditions_pos, self.grounding_preconditions_neg = self.get_grounding_preconditions()

    def get_masked_patterns(self, action_arity, predicate_arity):

        out_set = set()

        if predicate_arity <= 1 or action_arity <= 0:
            return set()

        for i in range(1, predicate_arity):
            if i + action_arity < predicate_arity:
                continue
            for blanks in itertools.combinations(range(predicate_arity), i):
                for action_args in itertools.permutations(range(action_arity), predicate_arity - i):
                    cur_arg_pos = 0
                    pattern_tuple = [None] * predicate_arity
                    for j in range(predicate_arity):
                        if not j in blanks:
                            pattern_tuple[j] = action_args[cur_arg_pos]
                            cur_arg_pos += 1
                    out_set.add(tuple(pattern_tuple))
        return out_set


    # get precondition patterns
    # todo need to check what this function is suppossed to do
    def get_grounding_preconditions(self):
        possible_positive_preconditions = dict()
        possible_negative_preconditions = dict()
        for action, a_arity in self.action_arity.items():
            possible_positive_preconditions[action] = dict()
            possible_negative_preconditions[action] = dict()
            for predicate, p_arity in self.predicate_arity_dict.items():

                if p_arity > 0:
                    possible_positive_preconditions[action][predicate] = set([pattern for pattern in
                                                                    itertools.product(range(a_arity), repeat=p_arity)])
                    possible_negative_preconditions[action][predicate] = set([pattern for pattern in
                                                                    itertools.product(range(a_arity), repeat=p_arity)])
                else:
                    possible_positive_preconditions[action][predicate] = {tuple()}
                    possible_negative_preconditions[action][predicate] = {tuple()}
        '''
        TODO check this
        '''

        all_predicates = set(self.predicate_arity_dict.keys())


        for state_num, state in enumerate(self.state_trace[:-1]):
            parsed_state = self.json.parse_state_precondition_test(state)
            # parsed_state_full = self.parse_state(state)
            cur_action = self.action_name_list[state_num]
            for predicate in all_predicates:

                if cur_action == '' or cur_action == 'none': continue

                if predicate not in parsed_state:
                    possible_positive_preconditions[cur_action][predicate] = set()
                    continue

                possible_positive_preconditions[cur_action][predicate] = self.get_mapping_patterns(
                    parsed_state[predicate],
                    self.action_object_list[state_num],
                    possible_positive_preconditions[cur_action][predicate],
                    True
                )

                possible_negative_preconditions[cur_action][predicate] = self.get_mapping_patterns(
                    parsed_state[predicate],
                    self.action_object_list[state_num],
                    possible_negative_preconditions[cur_action][predicate],
                    False
                )
                '''
                if predicate not in parsed_full_state:
                    masked_positive_preconditions[cur_action][predicate] = set()
                    continue
                masked_positive_preconditions[cur_action][predicate] = self.get_masked_preconditions(
                    parsed_full_state[predicate],
                    self.action_object_list[state_num],
                    masked_positive_preconditions[cur_action][predicate],
                    True
                )
                masked_negative_preconditions[cur_action][predicate] = self.get_masked_preconditions(
                    parsed_full_state[predicate],
                    self.action_object_list[state_num],
                    masked_negative_preconditions[cur_action][predicate],
                    False
                )
                '''
        return possible_positive_preconditions, possible_negative_preconditions

    def get_mapping_patterns(self, predicate_objects, action_objects, possible_mappings, positive: bool):

        new_possible_mappings = set()

        for mapping in possible_mappings:
            #if not len(mapping) <= len(action_objects):
            #    raise NotImplementedError
            #    continue
            possible = False
            for predicate_tuple in predicate_objects:
                possible = True
                for map_num, val in enumerate(mapping):
                    #if positive:
                    possible = True

                    if predicate_tuple[map_num] != action_objects[val]:
                        possible = False
                        break
                if possible: break
            if possible and positive:
                new_possible_mappings.add(mapping)
            elif not possible and not positive:
                new_possible_mappings.add(mapping)
        return new_possible_mappings

    def get_masked_preconditions(self, possible_predicates, action_objects, possible_masks, positive):

        raise NotImplementedError


    def get_precondition_string(self, act):

        prec_string = ''
        for pred in self.grounding_preconditions_pos[act]:
            for pat in self.grounding_preconditions_pos[act][pred]:
                prec_string += '\n\t\t\t\t ({} {})'.format(pred, ' '.join(f'?x{j}' for j in pat))

        for pred in self.grounding_preconditions_neg[act]:
            for pat in self.grounding_preconditions_neg[act][pred]:
                prec_string += '\n\t\t\t(not ({} {}))'.format(pred, ' '.join(f'?x{j}' for j in pat))

        return prec_string

    def get_effect_string(self, act, effects):

        eff_string = ''

        for sign in effects:
            if act not in effects[sign]:
                continue
            for pred in effects[sign][act]:
                for pattern in effects[sign][act][pred]:
                    if sign == 1:
                        eff_string += '\n\t\t\t\t ({} {})'.format(pred, ' '.join(f'?x{j}' for j in pattern))
                    elif sign == 0:
                        eff_string += '\n\t\t\t(not ({} {}))'.format(pred, ' '.join(f'?x{j}' for j in pattern))

        return eff_string

    def get_pddl_action_model(self):

        effects = self.get_effect_argument_positions()
        self.set_grounding_preconditions()

        pddl_model  = '(define (domain xyz)\n'
        pddl_model += '(:requirements :negative-preconditions)\n\n'
        pddl_model += '(:predicates \n'
        for pred, ar in self.predicate_arity_dict.items():
            pddl_model += '\t({} '.format(pred) +  ' '.join(f'?x{j}' for j in range(ar)) + ')\n'
        pddl_model += ')\n\n'

        for act, ar in self.action_arity.items():
            if act == '' or act == 'none':
                continue
            pddl_model += '(:action {}\n'.format(act)
            pddl_model += ' :parameters ({}) \n'.format(' '.join(f'?x{j}' for j in range(ar)))
            pddl_model += ' :precondition (and {})\n'.format(self.get_precondition_string(act))
            pddl_model += ' :effect (and {}\n\t\t  )\n'.format(self.get_effect_string(act, effects))
            pddl_model += ')\n\n'

        pddl_model += ')\n'
        print(pddl_model)