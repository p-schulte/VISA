import copy
import itertools
import time
from operator import contains
from typing import override

import pymimir
from py_separator_utils.mimir_things import mimir_thing
#from py_separator_utils.ZFeature import z_feature
#from py_separator_utils.negated_z_features import NegatedZFeature


class TraceIterator:

    def __init__(self, length, state_trace, action_trace):
        self.current = 0
        self.length = length
        self.state_trace = state_trace
        self.action_trace = action_trace

    def __next__(self):
        if self.current + 1 > self.length:
            raise StopIteration

        value = (self.state_trace[self.current], self.action_trace[self.current])
        self.current += 1
        return value

    def __iter__(self):
        return self


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


class Trace:

    def __init__(self, length: int, problem: mimir_thing, dropped_args: dict, predicate_arity: dict, dropped_pred: set):
        self.validation = False
        self.length = length
        self.current = 0
        self.problem = problem
        self.dropped_args = dropped_args
        self.dropped_preds = dropped_pred


        self.state_trace, self.action_trace = self.problem.sample_trace_from_init(self.length)
        self.action_name_list = [self.problem.get_action_name_by_index(action.get_action_index())
                                for action in self.action_trace]
        self.action_object_list = [action.get_object_indices() for action in self.action_trace]


        self.action_arity = {a_name: a_arity for (a_name, a_arity) in self.problem.get_action_arity()}

        self.hidden_action_arity = copy.deepcopy(self.action_arity)
        self.hidden_action_object_trace = copy.deepcopy(self.action_object_list)

        self.parsed_state_dict = dict()
        self.predicate_arity = predicate_arity

        self.predicate_arity_dict = {name: arity for (name,arity) in self.predicate_arity}

        if self.dropped_args is not None:
            for num, name in enumerate(self.action_name_list):
                if name in self.dropped_args:
                    self.action_object_list[num] = [val for pos, val in enumerate(self.action_object_list[num])
                                                     if pos not in self.dropped_args[name]]


            for name in self.dropped_args:
                self.action_arity[name] = self.action_arity[name] - len(dropped_args[name])

        # dictionary to store queries of appended arguments
        self.argument_queries = {action: {ar: None for ar in range(arity)}
                                 for action, arity in self.action_arity.items()}

        # for each action there is a set of effects
        self.affected_object_list, self.affected_tuple_list = [], []
        for action in self.action_trace:
            position_dict, tuple_dict = self.get_effects_of_action(action)
            self.affected_object_list.append(position_dict)
            self.affected_tuple_list.append(tuple_dict)

        # map action arguments to effected predicates, where no argument prediction is needed
        self.effect_mapping = set()

        self.candidate_dict = self.predicate_patterns()
        self.initialize_dict(self.candidate_dict)

        self.grounding_preconditions_pos, self.grounding_preconditions_neg, self.mask_pre_pos, self.mask_pre_neg = self.get_grounding_preconditions()
        self.validation = False

        self.predicate_types = None
        self.set_predicate_types()

    def __iter__(self):
        return CountUp(self.length)

    def set_predicate_types(self):
        self.predicate_types = self.problem.set_predicate_types_for_trace(self.state_trace)

    def get_predicate_types(self):
        return self.predicate_types

    def get_effects_of_action(self, action):

        out_dict = {0: dict(), 1:dict()}
        affected_tuple_dict = {0: dict(), 1:dict()}

        effatoms = dict()
        effatoms[1] = self.problem.pddl_repo.get_fluent_ground_atoms_from_indices(action.get_strips_effect().get_positive_effects())
        effatoms[0] = self.problem.pddl_repo.get_fluent_ground_atoms_from_indices(action.get_strips_effect().get_negative_effects())

        for effect, atoms in effatoms.items():
            for atom in atoms:

                name = atom.get_predicate().get_name()
                objs = [x.get_index() for x in atom.get_objects()]

                if name in self.dropped_preds:
                    continue

                if name in out_dict[effect]:
                    try:
                        for obj_pos, obj in enumerate(objs):
                            out_dict[effect][name][obj_pos].add(obj)
                        affected_tuple_dict[effect][name].add(tuple(objs))
                    except KeyError:
                        '''
                        if we are here the arity of a predicate changed what should never happen
                        '''
                        raise NotImplementedError
                else:
                    affected_tuple_dict[effect][name] = {(tuple(objs))}
                    out_dict[effect][name] = dict()
                    for obj_pos, obj in enumerate(objs):
                        out_dict[effect][name][obj_pos] = {obj}
        return out_dict, affected_tuple_dict

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

        r_val = self.problem.parse_state_with_dicts(copy.deepcopy(self.candidate_dict), position)
        self.parsed_state_dict[position] = r_val

        return r_val

    def initialize_dict(self, dicts):
        self.problem.initialize_state_with_statics(dicts)

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
            for position, query in self.argument_queries[action].items():
                print('The {}. argument of the action {} refers to the query {}'.format(position, action, query))

    def get_queries(self):
        return self.argument_queries

    '''
    we can make this 'simpler' with using the below function is_contained in position
    '''
    def add_arguments(self, argument_dict, action_name, query):

        if len(set(argument_dict.values())) == 1 and not self.validation:
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

    def check_final_args(self):

        missing_arguments = 0

        for action, arity in self.hidden_action_arity.items():
            for ar in range(arity):
                contains_object = False
                for pos in range(self.action_arity[action]):

                    contains_object = True

                    for position, action_name in enumerate(self.action_name_list):
                        if action != action_name:
                            continue
                        if self.hidden_action_object_trace[position][ar] != self.action_object_list[position][pos]:
                            contains_object = False
                            break

                    if contains_object:
                        break
                if contains_object:
                    continue
                else:
                    print("The objects in position {} of action {} are missing in the trace!".format(ar, action))
                    missing_arguments += 1

        if 0 == missing_arguments:
            print("\nTHERE ARE NO MISSING ARGUMENTS")

        all_correct = 0
        for action, arity in self.hidden_action_arity.items():

            if self.action_arity[action] > arity:
                all_correct += self.action_arity[action] - arity

                print("There are {} additional arguemnts for the action {}.".format(self.action_arity[action]-arity,
                                                                                    action))

        if all_correct == 0:
            print("There are no additional arguments")

        return missing_arguments, all_correct


    '''
        Get for each action the effects, 
        basicly only a dictionary is prepared
    '''
    def get_action_effects(self):
        action_effects = {0:dict(), 1: dict()}

        for action in self.problem.parsed_problem.get_domain().get_actions():
            action_effects[action.get_name()] = dict()
            for effect in action.get_strips_effect().get_effects():
                atom = effect.get_atom().get_predicate()
                if atom.get_name() in self.dropped_preds:
                    continue
                if effect.is_negated():

                    if action.get_name() not in action_effects[0]:
                        action_effects[0][action.get_name()] = dict()

                    action_effects[0][action.get_name()][atom.get_name()] = dict()

                    for pos in range(self.predicate_arity_dict[atom.get_name()]):

                        action_effects[0][action.get_name()][atom.get_name()][pos] = {i for i in range(
                        self.action_arity[action.get_name()])}

                else:

                    if action.get_name() not in action_effects[1]:
                        action_effects[1][action.get_name()] = dict()

                    action_effects[1][action.get_name()][atom.get_name()] = dict()

                    for pos in range(self.predicate_arity_dict[atom.get_name()]):

                        action_effects[1][action.get_name()][atom.get_name()][pos] = {i for i in range(
                        self.action_arity[action.get_name()])}

        return action_effects


    '''
        There can be some optimazation of the output, when there are more possible patterns than effects for an action
        that means, that there will be a minimal set of action patterns that will describe the effects, (if all 
        arguments contained in STRIPS like labels, this set will have the size of the effects ???)
    '''
    def get_effect_argument_positions(self):

        action_effects = self.get_action_effects()

        for position, action in enumerate(self.action_name_list):
            effects = self.affected_object_list[position]
            arguments = self.action_object_list[position]
            for sign in action_effects:
                if action not in action_effects[sign]:
                    continue
                for predicate in action_effects[sign][action]:
                    for predicate_position in action_effects[sign][action][predicate]:
                        for argument_position in list(action_effects[sign][action][predicate][predicate_position]):
                            if arguments[argument_position] not in effects[sign][predicate][predicate_position]:
                                action_effects[sign][action][predicate][predicate_position].remove(argument_position)


        pattern_dict = {0: dict(), 1:dict()}
        for sign in action_effects:
            for action in action_effects[sign]:
                pattern_dict[sign][action] = dict()
                for predicate in action_effects[sign][action]:
                    pattern_dict[sign][action][predicate] = set()
                    possible_argument_positions = []
                    for i in range(self.predicate_arity_dict[predicate]):
                        possible_argument_positions.append(action_effects[sign][action][predicate][i])
                    for pattern in itertools.product(*possible_argument_positions):
                        pattern_dict[sign][action][predicate].add(pattern)

        for position, action in enumerate(self.action_name_list):
            not_all_effects_covered = False
            for sign in pattern_dict:
                if action not in pattern_dict[sign]:
                    continue
                for predicate in pattern_dict[sign][action]:
                    # check whether each pattern has an corresponding effect
                    groundings_from_patterns = set()
                    for pattern in list(pattern_dict[sign][action][predicate]):
                        pattern_grounding = tuple([self.action_object_list[position][pattern_pos] for pattern_pos in pattern])

                        if pattern_grounding not in self.affected_tuple_list[position][sign][predicate]:
                            pattern_dict[sign][action][predicate].remove(pattern)
                        else:
                            groundings_from_patterns.add(pattern_grounding)
                    # check whether for each effect there is a pattern_grounding
                    if self.affected_tuple_list[position][sign][predicate] != groundings_from_patterns:
                        not_all_effects_covered = True
            if not_all_effects_covered:
                print("For an effect of the action {} there arguments of effects missing".format(
                    action
                ))

        self.effect_mapping = pattern_dict
        return pattern_dict

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

    def set_grounding_preconditions(self):
        self.grounding_preconditions_pos, self.grounding_preconditions_neg, self.mask_pre_pos, self.mask_pre_neg = self.get_grounding_preconditions()

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



    def get_grounding_preconditions(self):
        possible_positive_preconditions = dict()
        possible_negative_preconditions = dict()
        masked_positive_preconditions = dict()
        masked_negative_preconditions = dict()
        for action, a_arity in self.action_arity.items():
            possible_positive_preconditions[action] = dict()
            possible_negative_preconditions[action] = dict()
            masked_negative_preconditions[action] = dict()
            masked_positive_preconditions[action] = dict()
            for predicate, p_arity in self.predicate_arity_dict.items():

                if predicate in self.dropped_preds:
                    continue

                if p_arity > 0:
                    possible_positive_preconditions[action][predicate] = set([pattern for pattern in
                                                                    itertools.permutations(range(a_arity), r=p_arity)])
                    possible_negative_preconditions[action][predicate] = set([pattern for pattern in
                                                                              itertools.permutations(range(a_arity),
                                                                                                     r=p_arity)])
                else:
                    possible_positive_preconditions[action][predicate] = {tuple()}
                    possible_negative_preconditions[action][predicate] = {tuple()}

                if p_arity >= 2:
                    masked_negative_preconditions[action][predicate] = self.get_masked_patterns(a_arity, p_arity)
                    masked_positive_preconditions[action][predicate] = self.get_masked_patterns(a_arity, p_arity)
        '''
        TODO check this
        '''

        all_predicates = set(self.predicate_arity_dict.keys())


        for state_num, state in enumerate(self.state_trace[:-1]):
            parsed_state, parsed_full_state = self.problem.parse_state_precondition_test(state)
            # parsed_state_full = self.parse_state(state)
            cur_action = self.action_name_list[state_num]
            for predicate in all_predicates:
                if predicate in self.dropped_preds:
                    continue
                #if predicate == 'empty-ferry':
                #    print('test')
                if predicate not in parsed_state:
                    possible_positive_preconditions[cur_action][predicate] = set()
                    masked_positive_preconditions[cur_action][predicate] = set()
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

        #for action in possible_negative_preconditions:
        #    print(action,'neg', possible_negative_preconditions[action])
        #    print(action,'pos', possible_positive_preconditions[action])


        #for action in self.action_arity:
        #    print('Possible preconditions of the action {} are: {}'.format(action,
        #                                                                   possible_positive_preconditions[action]))
        return possible_positive_preconditions, possible_negative_preconditions, masked_positive_preconditions, masked_negative_preconditions

    def get_mapping_patterns(self, predicate_objects, action_objects, possible_mappings, positive: bool):

        new_possible_mappings = set()

        for mapping in possible_mappings:
            if not len(mapping) <= len(action_objects):
                continue
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
                    #else:
                    #    if predicate_tuple[map_num] == action_objects[val]:
                    #        possible = True
                    #        break
            if possible and positive:
                new_possible_mappings.add(mapping)
            elif not possible and not positive:
                new_possible_mappings.add(mapping)
        return new_possible_mappings

    def get_masked_preconditions(self, possible_predicates, action_objects, possible_masks, positive):

        still_possible_masks = set()

        for masked in possible_masks:
            mask = tuple([None if m is None else 0 for m in masked])
            masked_grounding = tuple([None if pos is None else action_objects[pos] for pos in masked])
            if masked_grounding in possible_predicates[mask] and positive:
                still_possible_masks.add(masked)
            elif not positive and masked_grounding not in possible_predicates[mask]:
                still_possible_masks.add(masked)

        return still_possible_masks

    def to_json_format(self):
        outl = []
        for i in range(1, self.length):
            cur_dict = dict()
            cur_dict['action_name'] = self.action_name_list[i]
            cur_dict['action_objects'] = self.action_object_list[i]

            state = self.problem.parse_state_for_json(self.state_trace[i])

            json_state = {feat: [] for feat in self.predicate_arity_dict.keys() if feat != 'eq'}
            for feat in state:
                for gr in state[feat]:
                    json_state[feat].append(list(gr))

            cur_dict['state'] = json_state

            outl.append(cur_dict)
        return outl



class ValidationTrace(Trace):

    def __init__(self, problem: mimir_thing, dropped_args: dict, predicate_arity: dict, dropped_pred: set, state,
                 predicate_types):
        super().__init__(1, problem, dropped_args, predicate_arity, dropped_pred)
        self.affected_object_list, self.affected_tuple_list = None, None
        self.validation = True
        self.state_trace[0] = state
        self.reset_parsed_dict()
        self.predicate_types = predicate_types


    def set_state(self, state_index):
        self.state_trace[0] = state_index

    def set_action(self, name, objects):
        self.action_name_list[0] = name
        self.action_object_list[0] = objects

    def get_all_applicable_actions(self, state_index):
        possible_actions = dict()
        all_actions = self.problem.get_all_applicable_actions(state_index)
        for action in all_actions:

            action_name = self.problem.get_action_name_by_index(action.get_action_index())
            action_objects = action.get_object_indices()
            if self.dropped_args is not None and action_name in self.dropped_args:
                args = tuple([obj for obj_num, obj in enumerate(action_objects)
                             if obj_num not in self.dropped_args[action_name]])
                possible_actions[(action_name, args)] = self.get_effects_of_action(action)[1]
            else:
                possible_actions[(action_name, tuple(action_objects))] = self.get_effects_of_action(action)[1]

        return possible_actions

    def get_full_grounding(self):
        return self.action_object_list[0]

    @override
    def set_predicate_types(self):
        self.predicate_types = None

    '''
    def validation(self, queries, effect_mapping):
        print("help")
        Action_candidate_dict = dict()
        for action, positions in queries.items():

            action_z_features = []

            for position in positions:
                if queries[action][position] is None:
                    continue
                for (predicate, pattern) in queries[action][position]:
                    if len(predicate) > 4 and predicate[:4] == "not-":
                        action_z_features.append(NegatedZFeature(pattern, predicate[4:], action,
                                                                 self.problem.predicate_types,
                                                                 self.problem.objects_types))

                    else:
                        action_z_features.append(z_feature(pattern, predicate, action, self.predicate_arity_dict))

            Action_candidate_dict[action] = ActionCandidates(action,None, None, self.predicate_arity, 0, self.problem.predicate_types,
                             self.problem.objects_types, action_z_features)

        for pos, action in enumerate(self.action_name_list):
            Action_candidate_dict[action].parse_state(self.parse_state(pos), self.action_object_list[pos], pos)
    '''
