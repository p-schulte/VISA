import copy
import itertools
import json
import time
import pymimir
import io
import argparse
import os
from contextlib import redirect_stdout
from graph_generator import create_random_initial_state
from py_separator_utils.json_things import JSONThings
from py_separator_utils.trace_json import JSONTrace
from py_separator_utils.mimir_things import mimir_thing
from py_separator_utils.ActionAdds import ActionCandidates, AllActionCandidates, ValidationFailed, OverallValidationFailed
from py_separator_utils.ZFeature import z_feature
from py_separator_utils.negated_z_features import NegatedZFeature
from py_separator_utils.trace import Trace, ValidationTrace
from pathlib import Path
from itertools import chain, combinations, groupby


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def get_this_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--domain", type=str, required=False,
                        help="specify domain that is in the pddl_files folder.")
    parser.add_argument("-i", "--instance", type=str, required=False,
                        help="specify list of instances that is in the pddl_files folder.")
    parser.add_argument('-a', '--dropped_args', type=Path, required=False,
                        help='Defines arguments that will be deleted')
    parser.add_argument('-p', '--dropped_predicates', type=Path, required=False,
                        help='Defines arguments that will be deleted')
    parser.add_argument("-s", "--size", type=int, required=False,
                        help="Length of the sampled trace.")

    parser.add_argument("-t", "--testall", required=False, action="store_true",
                        help="Runs all test that are defined somewhere")
    parser.add_argument('--drop-test', required=False, action="store_true",
                        help='Check which arguments can be dropped without losing information')
    parser.add_argument('-r', '--reduce_strips', action='store_true',
                        help='reduces a strips domain to strips plus with an action trace.')
    parser.add_argument('-v', '--validation_instance', required=False, type=str,
                        help='Instance on which the learned domain is verified.')
    parser.add_argument('--test_all_dropped', required=False, action='store_true',
                        help='Do all tests with dropped arguments')
    parser.add_argument('-si','--single_test', required=False, action='store_true',
                        help='Do all tests with dropped arguments')
    parser.add_argument('--test_drop', required=False, action='store_true',
                        help='Do all tests with dropped predicates')
    parser.add_argument('--run', required=False, type=int,
                        help='Number of current_run')
    parser.add_argument('-vs', '--validation_size', required=False, type=int,
                        help='Size of partial graph used for verification')
    parser.add_argument('-n', '--name', type=str, required=False,
                        help='Name of Domain.')
    parser.add_argument('--json', required=False, type=str,
                        help='Calls the algorithm on the given json file!')
    parser.add_argument('--json_add_static', required=False, action='store_true',
                        help='Adds static arguments!')

    return parser.parse_args()

def get_dropped_dict(file_path):
    data = dict()
    with open(file_path, 'r') as file:
        for line in file:
            if len(line) < 2 or line[0] == "#":
                continue
            split_line = line.strip()
            split_line = split_line.split(' ')
            data[split_line[0]] = set()
            for arg in split_line[1:]:
                data[split_line[0]].add(int(arg))

    print(data)

    return data

def get_dropped_preds(file_path):

    if file_path is None:
        return set()
    elif isinstance(file_path, set):
        return file_path

    data = set()
    with open(file_path, 'r') as file:
        for line in file:
            if len(line) < 2 or line[0] == "#":
                continue
            split_line = line.strip()
            split_line = split_line.split(' ')
            data.add(split_line[0])

    return data

def print_effects(effects):
    for sign in effects:
        if sign == 0:
            print("Positive Effects: \n")
        else:
            print('\n Negative Effects: \n')
        for action in effects[sign]:
            for predicate in effects[sign][action]:
                for pattern in effects[sign][action][predicate]:
                    print('{}{} on {}'.format(action, pattern, predicate))


def get_effects_for_grounding(effects, action, grounding):
    out = dict()
    for sign in effects:
        if action not in effects[sign]:
            continue
        out[sign] = dict()
        for predicate in effects[sign][action]:
            out[sign][predicate] = set()
            for pattern in effects[sign][action][predicate]:
                effected_grounding = [grounding[pos] for pos in pattern]
                out[sign][predicate].add(tuple(effected_grounding))
    return out


def covers_all_variables(rule_dict, init: set, all_positions):
    current = init
    while True:
        next = copy.deepcopy(current)
        for ps in powerset(current):
            try:
                next.update(rule_dict[frozenset(ps)])
            except KeyError:
                continue
        if all_positions == next:
            return True
        elif  len(next) == len(current):
            return False
        current = next


def get_patterns(predicate, pattern):
    out_set = set()
    for pos in range(len(pattern)):
        other_positions = {x for x in range(len(pattern)) if x != pos}
        for subset in powerset(other_positions):
            cur_pattern = tuple([
                -1 if pos == cur_pos
                else pattern[cur_pos] if cur_pos in subset
                else None
                for cur_pos in range(len(pattern))
            ])
            out_set.add((predicate, cur_pattern))
    return out_set

def reduce_strips(domain, instance, size, dropped_predicates):
    action_schema1 = strips_to_strips_plus(domain, instance, size, dropped_predicates,True)
    action_schema2 = strips_to_strips_plus(domain, instance, size, dropped_predicates, False)

    print('Action Schema with only PRECONDITION pattern:\n', action_schema1)
    print('Action Schema with All pattern:\n', action_schema2)

    is_smaller, is_larger = False, False

    for action, arity in action_schema1.items():
        if action_schema2[action] < arity:
            is_smaller = True
        elif action_schema2[action] > arity:
            is_larger = True

    if is_smaller:
        print('The Action Schema with all patterns is smaller than the action schema using only precondition pattern!')
    if is_larger:
        raise ValidationFailed
    return is_smaller, sum([val for _, val in action_schema1.items()]), sum([val for _, val in action_schema2.items()])


def check_simple_preconditions(val_state, preconditions, object_tuple, positive):
    possible = True
    for prec_predicate in preconditions:
        for prec_pattern in preconditions[prec_predicate]:
            prec_grounding = tuple([object_tuple[pos] for pos in prec_pattern])
            if positive:
                if prec_predicate not in val_state:
                    possible = False
                    break
                if prec_grounding not in val_state[prec_predicate]:
                    possible = False
                    break
            else:
                if prec_predicate not in val_state:
                    continue
                if prec_grounding in val_state[prec_predicate]:
                    possible = False
                    break
        if not possible:
            break
    return possible


def strips_to_strips_plus(domain, instance, size, dropped_predicates, use_only_preconditions):

    dropped_p = get_dropped_preds(dropped_predicates)
    planning_problem = mimir_thing(domain, instance, dropped_p, None)
    trace = Trace(size, planning_problem, dict(), planning_problem.get_feature_arity(), dropped_p)

    if use_only_preconditions:

        precondition_patterns = planning_problem.get_preconditions()
        print(precondition_patterns)
        possible_features = {action: set() for action in precondition_patterns}
        for action, preconditions in precondition_patterns.items():
            for pre in preconditions:
                possible_features[action].update(get_patterns(pre[0], pre[1]))
            print(possible_features[action])

        action_candidate_dict = dict()
        for action, feature in possible_features.items():
            action_z_features = []
            for (predicate, pattern) in feature:
                if len(predicate) > 4 and predicate[:4] == "not-":
                    action_z_features.append(NegatedZFeature(pattern, predicate[4:], action,
                                                             trace.predicate_types,
                                                             trace.problem.objects_types))
                else:
                    action_z_features.append(z_feature(pattern, predicate, action, trace.predicate_arity_dict))

            action_candidate_dict[action] = action_z_features

        candidates = AllActionCandidates(trace.action_arity, planning_problem.get_feature_arity(),
                                         trace.get_predicate_types(), planning_problem.get_object_types(),
                                         action_candidate_dict)
    else:
        candidates = AllActionCandidates(trace.action_arity, planning_problem.get_feature_arity(),
                                         trace.get_predicate_types(), planning_problem.get_object_types(),
                                         None)

    #candidates = AllActionCandidates(trace.action_arity, planning_problem.get_feature_arity(),
    #                                 planning_problem.get_predicate_types(), planning_problem.get_object_types(),
    #                                 action_candidate_dict)

    for t in trace:
        parsed_state = trace.parse_state(trace.get_state_of_index(t))
        candidates.parse_state(parsed_state, trace.get_action_name(t), trace.get_action_objects(t), t)

    # these functions should be changed, s.t. we get the things that were not able to be added
    quries_and_positions = candidates.get_not_addable_queries(trace)

    print(quries_and_positions)
    needed_positions = dict()

    action_arity_dict = {action: arity for (action, arity) in planning_problem.action_arity}

    result_dict = dict()
    for action, queries in quries_and_positions.items():
        rule_dict = dict()
        for query, position in queries.items():
            if isinstance(query, tuple):
                query_arguments = frozenset([x for x in query[1] if isinstance(x, int) and x > -1])
            else:
                query_arguments = frozenset([x for q in query for x in q[1] if isinstance(x, int) and x > -1])
            print("contained_args",query_arguments, query)
            if query_arguments in rule_dict:
                rule_dict[query_arguments].update(position)
            else:
                rule_dict[query_arguments] = position

        all_positions = {x for x in range(action_arity_dict[action])}

        found = False

        for i in range(action_arity_dict[action]):
            solutions = set()
            if i == 0:
                possible = covers_all_variables(rule_dict, set(), all_positions)
                if possible:
                    solutions.add(frozenset())
            else:
                for init in itertools.combinations(all_positions, r=i):
                    possible = covers_all_variables(rule_dict, set(init), all_positions)
                    if possible:
                        solutions.add(frozenset(init))

            if len(solutions) > 0:
                print("For action {} there are at least {} arguments needs, which can be {}".format(
                    action, i, [tuple(x) for x in solutions]
                ))
                result_dict[action] = i
                found = True

                cur_solution = list(solutions)[0]
                needed_positions[action] = set([x for x in range(action_arity_dict[action]) if x not in cur_solution])

                break
        if not found:

            result_dict[action] = action_arity_dict[action]
            print("For action {} there are at least {} arguments needs, which can be {}".format(
                action, action_arity_dict[action], [tuple([x for x in range(action_arity_dict[action])])]
            ))

    print(result_dict)
    return needed_positions


def get_contained_z_features(all_unique_queries, current_action, validation_trace, validation_problem):
    action_candidate_dict = dict()

    # Build z-features needed for this.
    for action, queries in all_unique_queries.items():

        if action != current_action:
            continue

        action_z_features = []
        for queryqq in queries:
            (_, query) = queryqq
            for (predicate, pattern) in query:
                if len(predicate) > 4 and predicate[:4] == "not-":
                    action_z_features.append(NegatedZFeature(pattern, predicate[4:], action,
                                                             validation_trace.predicate_types,
                                                             validation_trace.problem.objects_types))
                else:
                    action_z_features.append(
                        z_feature(pattern, predicate, action, validation_problem.feature_arity_dict))

        action_candidate_dict[action] = action_z_features
    return action_candidate_dict


def get_contained_z_features_adding_queries(all_unique_queries, current_action, validation_trace, validation_problem):
    action_candidate_dict = dict()

    # Build z-features needed for this.
    for action, queries in all_unique_queries.items():

        if action != current_action:
            continue

        action_z_features = []
        for position, query in queries.items():

            if query is None:
                continue

            for (predicate, pattern) in query:
                if len(predicate) > 4 and predicate[:4] == "not-":
                    action_z_features.append(NegatedZFeature(pattern, predicate[4:], action,
                                                             validation_trace.predicate_types,
                                                             validation_trace.problem.objects_types))
                else:
                    action_z_features.append(
                        z_feature(pattern, predicate, action, validation_problem.feature_arity_dict))

        action_candidate_dict[action] = action_z_features
    return action_candidate_dict

def check_masked_preconditions(masked_preconditions, val_masked_pre, current_action, object_grounding, positive,
                               validation_trace):
    for pred in masked_preconditions[current_action]:
        if pred not in val_masked_pre:
            if positive and len(masked_preconditions[current_action][pred]) > 0:
                print('Failed here1')
                return False
            continue
        new_precs = validation_trace.get_masked_preconditions(val_masked_pre[pred],
                                                              object_grounding,
                                                              masked_preconditions[
                                                              current_action][pred],
                                                              positive)
        if len(new_precs) < len(masked_preconditions[current_action][pred]):
            print('Failed here2')
            return False
    return True

def find_additional_arguments(domain, instance, size, dropped_args, dropped_predicates, validation_instance, val_size):
    
    time_start = time.time()

    if dropped_predicates:
        dropped_p = get_dropped_preds(dropped_predicates)
    else:
        dropped_p = set()

    da = strips_to_strips_plus(domain, instance, size, set(), True)

    if val_size is None:
        val_size = 300

    if validation_instance is None:
        validation_instance = instance

    x = mimir_thing(domain, instance, dropped_p, da)

    num_objects = len(x.parsed_problem.get_problem().get_objects())

    #if dropped_args:
    #    da = get_dropped_dict(dropped_args)
    #else:
    #    da = None

    #print('Reduced Schema', reduced_schema)
    print("dropped Args", da)
    predicate_arities = x.get_feature_arity()



    object_set = x.get_object_set()

    '''
        TODO dropped predicates need to be removed from this dict
    '''
    new_Trace = Trace(size, x, da, predicate_arities, dropped_p)

    json_dump = new_Trace.to_json_format()

    with open('json_files/blocksworld.json', 'w') as f:
        json.dump(json_dump, f ,indent=4)
 

    return None
    initial_arities = copy.deepcopy(new_Trace.action_arity)

    preconditions_of_groundings = copy.deepcopy(new_Trace.grounding_preconditions_pos)
    neg_preconditions_of_groundings = copy.deepcopy(new_Trace.grounding_preconditions_neg)

    print("possible preconditions", preconditions_of_groundings)

    num_initial_args = sum([ar for action, ar in new_Trace.action_arity.items()])
    new_all_things = None

    i = 0
    while True:

        new_Trace.reset_parsed_dict()
        i += 1
        print("This is the {}. iteration".format(i))
        new_all_things = AllActionCandidates(new_Trace.action_arity, predicate_arities, new_Trace.get_predicate_types(),
                                             x.get_object_types(), None)

        for t in new_Trace:
            new_parsed_state = new_Trace.parse_state(new_Trace.get_state_of_index(t))
            new_all_things.parse_state(new_parsed_state, new_Trace.get_action_name(t), new_Trace.get_action_objects(t), t)

        was_there_somehting_added = new_all_things.add_arguments(new_Trace)

        '''
            for combis all arguments should be unique, else there can not be a precondtion on the combination
            this would lead to problems when defining the domain since we would need to derive predicates 
            that are not in the domain and the corresponding precondition can not be stated
            
            why in npuzzle every argument is found? CRISP description...
        '''
        combi_added = new_all_things.check_combis(new_Trace)


        if not was_there_somehting_added and not combi_added:
        #if not was_there_somehting_added:
            break

    effects = new_Trace.get_effect_argument_positions()

    print_effects(effects)

    time_end = str(round(time.time() - time_start, 2)) + ' s'
    
    num_missing, num_additional = new_Trace.check_final_args()
    new_Trace.print_action_arity()
    
    num_domain_args = sum([ar for act, ar in new_Trace.hidden_action_arity.items()])
    num_learned_args = sum([ar for action, ar in new_Trace.action_arity.items()])
    
    out = [num_domain_args, num_initial_args, num_learned_args, num_missing, num_additional, time_end, size, num_objects]

    new_Trace.print_query_output()

    # new_all_things.set_unique_patterns()
    all_unique_queries = new_all_things.get_unique_queries()
    adding_queries = new_Trace.get_queries()
    effect_mapping = new_Trace.get_effect_mapping()

    print('\n-----------------------------\n')

    print('%%%%%%   Adding Querries   %%%%')
    for a, q in adding_queries.items():
        for qq in q:
            print(a, qq)

    print('\n %%%%%% Unique quries %%%%%%')
    for a, q in all_unique_queries.items():
        for qq in q:
            print(a, qq)

    print('\n-----------------------------\n')

    new_Trace.set_grounding_preconditions()
    preconditions_of_groundings_all = copy.deepcopy(new_Trace.grounding_preconditions_pos)
    neg_preconditions_of_groundings_all = copy.deepcopy(new_Trace.grounding_preconditions_neg)
    masked_all_positive_preconditions = copy.deepcopy(new_Trace.mask_pre_pos)
    masked_all_negative_preconditions = copy.deepcopy(new_Trace.mask_pre_neg)


    '''
        Verification
    '''
    validation_time_start = time.time()
    seen_actions = {act: False for act in new_Trace.action_arity}

    test_set = {(act, val) for act in new_Trace.action_arity for val in {True, False}}

    validation_problem = mimir_thing(domain, validation_instance, dropped_p, da)
    validation_problem.sample_bfs_from_init(300)

    validation_counter = 0

    for (current_action, positive_verification) in test_set:
        print('current_action: ', current_action)

        for _ in range(200):
            (random_node, cur_object_tuple, old_effects) = validation_problem.get_random_state_action_pair(positive_verification,
                                                                                              current_action)
            if random_node is None and cur_object_tuple is None:
                break

            validation_counter += 1

            print('state, object_tuple :', random_node, cur_object_tuple)

            validation_trace = ValidationTrace(validation_problem, da, predicate_arities, dropped_p, random_node,
                                               new_Trace.predicate_types)
            validation_trace.set_action(current_action, list(cur_object_tuple))

            try:
                action_candidate_dict = dict()

                # Build z-features needed for this.
                #action_candidate_dict = get_contained_z_features(all_unique_queries,
                #                                                 current_action,
                #                                                 validation_trace,
                #                                                 validation_problem)

                action_candidate_dict = get_contained_z_features_adding_queries(adding_queries,
                                                                 current_action,
                                                                 validation_trace,
                                                                 validation_problem)

                already_added = {current_action: 0}

                # add arguments to the action groundings
                while True:

                    all_validation_candidates = AllActionCandidates(validation_trace.action_arity,
                                                                    validation_trace.predicate_arity,
                                                                    validation_trace.get_predicate_types(),
                                                                    validation_problem.objects_types,
                                                                    action_candidate_dict)

                    for t in validation_trace:
                        state = validation_trace.parse_state(validation_trace.get_state_of_index(t))
                        all_validation_candidates.parse_state_validation(state, validation_trace.get_action_name(t),
                                                                         validation_trace.get_action_objects(t), t)

                    changed, all_found = False, True
                    # this loop must only be doen for the current action
                    for action, positions in adding_queries.items():
                        if action != current_action or positions == set() or positions == dict():
                            continue

                        # Adding additional arguments of z-features
                        while True:
                            current = already_added[action]
                            if current > max(positions):
                                break
                            if adding_queries[action][current] is None:
                                already_added[action] += 1
                            else:
                                added = False
                                try:
                                    added = all_validation_candidates.add_query_arguments(action,
                                                                                          adding_queries[action][current],
                                                                                          validation_trace)
                                except ValidationFailed:
                                    break
                                if added:
                                    changed = True
                                    already_added[action] += 1
                                else:
                                    break

                        all_found = True
                        for act, added in already_added.items():
                            if len(adding_queries[act]) > added and act == current_action:
                                all_found = False
                                break

                        if all_found:
                            break
                        elif not changed:
                            print("raise1")
                            raise ValidationFailed
                    if all_found:
                        break

                # should never be raised
                if not all_found:
                    print("raise2")
                    raise ValidationFailed

                all_validation_candidates = AllActionCandidates(validation_trace.action_arity,
                                                                validation_trace.predicate_arity,
                                                                validation_trace.predicate_types,
                                                                validation_problem.objects_types,
                                                                action_candidate_dict)

                for t in validation_trace:
                    state = validation_trace.parse_state(validation_trace.get_state_of_index(t))
                    all_validation_candidates.parse_state_validation(state, validation_trace.get_action_name(t),
                                                                     validation_trace.get_action_objects(t), t)

                # check if all queries from the original domain are also fulfilled in the learned domain
                #if not all_validation_candidates.check_all_unique_queries(all_unique_queries, validation_trace):
                #    print("raise3")
                #    raise ValidationFailed

                val_state = validation_problem.parse_state_precondition(validation_trace.state_trace[0])

                full_object_grounding = validation_trace.get_full_grounding()
                preconditions = preconditions_of_groundings_all[current_action]
                neg_preconditions = neg_preconditions_of_groundings_all[current_action]

                validation_trace.set_grounding_preconditions()
                _, val_masked_pre = validation_problem.parse_state_precondition_test(
                    validation_trace.state_trace[0])

                if not check_masked_preconditions(masked_all_positive_preconditions, val_masked_pre, current_action,
                                                  full_object_grounding, True, validation_trace):
                    print("raise4")
                    raise ValidationFailed

                if not check_masked_preconditions(masked_all_negative_preconditions, val_masked_pre, current_action,
                                                  full_object_grounding, False, validation_trace):
                    print("raise5")
                    raise ValidationFailed

                if not check_simple_preconditions(val_state, preconditions, full_object_grounding, True):
                    raise ValidationFailed

                if not check_simple_preconditions(val_state, neg_preconditions, full_object_grounding, False):
                    print("raise5")
                    raise ValidationFailed

                if not positive_verification:
                    print('The negative action {} with arguments {} was not verified!'.format(current_action,
                                                                                              cur_object_tuple))
                    print("raise6")
                    raise OverallValidationFailed



                new_eff = get_effects_for_grounding(effects, current_action, full_object_grounding)

        
                print('New Effects', new_eff)

        
                for sign in old_effects:
                    for predicate in old_effects[sign]:
                        if old_effects[sign][predicate] != new_eff[sign][predicate]:
                            raise OverallValidationFailed
        
                for sign in new_eff:
                    for predicate in new_eff[sign]:
                        if old_effects[sign][predicate] != new_eff[sign][predicate]:
                            raise OverallValidationFailed

            except ValidationFailed:
                if positive_verification:
                    print('The positive action {} with arguments {} was not verified!'.format(current_action, cur_object_tuple))
                    raise OverallValidationFailed
                #print('\n$$$$$$ All Unique Queries $$$$$$$')
                #for x in all_unique_queries[current_action]:
                #    print(x)
                #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
                #print('Validation Failed!!!', current_action, cur_object_tuple)

    validation_time_end = time.time() - validation_time_start
    time_end = str(round(time.time() - time_start, 2)) + ' s'
    out.append(time_end)

    out.append(validation_counter)

    out.append(len(validation_problem.parsed_problem.get_problem().get_objects()))
    print(time_end)

    print('%%%%%%   Adding Querries   %%%%')
    for q in adding_queries:
        print(q)

    print('\n %%%%%% Unique quries %%%%%%')
    for q in all_unique_queries:
        print(q)

    return out

    while True:
        validation_problem = mimir_thing(domain, validation_instance, dropped_p, da)
        validation_problem.sample_bfs_from_init(300)

        '''
         ToDo sample also a random action that is either applicable or not applicable
        '''
        random_node = validation_problem.get_random_state()
        (state_index, action_grounding) = validation_problem.get_random_state_action_pair(True, 'stack')
        validation_trace = ValidationTrace(validation_problem, da, predicate_arities, dropped_p, random_node, new_Trace.predicate_types)
        print("OBJECT-DICT @@@@@@@@@@@@@@@@@@@@@@@@@@\n", validation_problem.object_dict)

        '''
        This becomes unnecessary
        '''
        applicable_actions_with_effects = validation_trace.get_all_applicable_actions(random_node)
        applicable_action_set = set([x for x in applicable_actions_with_effects])

        for act in applicable_actions_with_effects:
            seen_actions[act[0]] = True

        res = set()
        for current_action, current_arity in initial_arities.items():

            #if current_action != 'fly':
            #    continue

            test_counter = 0

            val_state = validation_problem.parse_state_precondition(validation_trace.state_trace[0])
            preconditions = preconditions_of_groundings[current_action]
            possible_objects = [copy.deepcopy(set([obj for obj in validation_problem.object_dict]))] * current_arity


            #if not check_simple_preconditions(val_state, preconditions, cur_object_tuple, True):
            #    continue

            for prec_predicate in preconditions:
                for prec_pattern in preconditions[prec_predicate]:
                    for pos_num, pos in enumerate(prec_pattern):
                        position_arguments = set()
                        if prec_predicate not in val_state:
                            continue
                        for prec_grounding in val_state[prec_predicate]:
                            position_arguments.add(prec_grounding[pos_num])
                        possible_objects[pos] = possible_objects[pos].intersection(position_arguments)


            if current_arity > 1:
                if set() in possible_objects:
                    continue
                all_combis = [x for x in itertools.product(*possible_objects)]
            elif current_arity == 1:
                all_combis = [tuple([x]) for x in possible_objects[0]]
            else:
                all_combis = [tuple()]

            for cur_object_tuple in all_combis:

                print('\n New Combi\n')

                #if cur_object_tuple != (6, 13):
                #    continue
                #if cur_object_tuple not in {(0,0), (1,1), (2,2), (3,3), (4,4)}:
                #    continue

                test_object_tuple = copy.deepcopy(cur_object_tuple)

                #test_counter += 1
                #if test_counter > 200:
                #    break

                preconditions = preconditions_of_groundings[current_action]
                neg_preconditions = neg_preconditions_of_groundings[current_action]
                # HARDCODED THE POSITION

                if not check_simple_preconditions(val_state, preconditions, cur_object_tuple, True):
                    continue

                if not check_simple_preconditions(val_state, neg_preconditions, cur_object_tuple, False):
                    continue

                '''
                possible = True
                for prec_predicate in preconditions:
                    for prec_pattern in preconditions[prec_predicate]:
                        prec_grounding = tuple([cur_object_tuple[pos] for pos in prec_pattern])
                        if prec_predicate not in val_state:
                            continue
                        if prec_grounding not in val_state[prec_predicate]:
                            possible = False
                            break
                    if not possible:
                        break

                if not possible:
                    continue

                #if current_action != 'stack':
                #    continue

                #if cur_object_tuple not in {(1,1), (2,2), (3,3), (4,4)}:
                #    continue

                for prec_predicate in neg_preconditions:
                    for prec_pattern in neg_preconditions[prec_predicate]:
                        prec_grounding = tuple([cur_object_tuple[pos] for pos in prec_pattern])
                        if prec_predicate not in val_state:
                            continue
                        if prec_grounding in val_state[prec_predicate]:
                            possible = False
                            break
                    if not possible:
                        break

                if not possible:
                    continue
                '''
                validation_trace = ValidationTrace(validation_problem, da, predicate_arities, dropped_p,random_node,
                                                   validation_trace.predicate_types)
                validation_trace.set_action(current_action, list(cur_object_tuple))

                #print(current_action, current_arity, cur_object_tuple)

                #print("test action", current_action, validation_trace.get_action_objects(0), cur_object_tuple)
                #print("arities", validation_trace.action_arity, initial_arities)

                '''
                    Old implementation
                '''
                try:
                    action_candidate_dict = dict()

                    # get all z-features contained in some queries
                    for action, queries in all_unique_queries.items():

                        if action != current_action:
                            continue

                        action_z_features = []
                        for queryqq in queries:
                            (_, query) = queryqq
                            for (predicate, pattern) in query:
                                if len(predicate) > 4 and predicate[:4] == "not-":
                                    action_z_features.append(NegatedZFeature(pattern, predicate[4:], action,
                                                                             validation_trace.predicate_types,
                                                                             validation_trace.problem.objects_types))
                                else:
                                    action_z_features.append(z_feature(pattern, predicate, action, validation_problem.feature_arity_dict))

                        action_candidate_dict[action] = action_z_features

                    already_added = {current_action: 0}

                    #if current_action != 'load' or cur_object_tuple != (3,7,13):
                    #    continue

                    # add arguments to the action groundings
                    while True:

                        print('Iteration X', validation_trace.action_arity)

                        all_validation_candidates = AllActionCandidates(validation_trace.action_arity,
                                                                        validation_trace.predicate_arity,
                                                                        validation_trace.get_predicate_types(),
                                                                        validation_problem.objects_types,
                                                                        action_candidate_dict)

                        for t in validation_trace:
                            state = validation_trace.parse_state(validation_trace.get_state_of_index(t))
                            all_validation_candidates.parse_state_validation(state, validation_trace.get_action_name(t),
                                                                             validation_trace.get_action_objects(t), t)

                        changed, all_found = False, True

                        for action, positions in adding_queries.items():

                            if positions == set() or positions == dict():
                                continue

                            if action != current_action:
                                continue

                            while True:
                                current = already_added[action]
                                print(positions)
                                if current > max(positions):
                                    break
                                if adding_queries[action][current] is None:
                                    already_added[action] += 1
                                else:
                                    added = False
                                    try:
                                        added = all_validation_candidates.add_query_arguments(action, adding_queries[action][current],
                                                                                              validation_trace)
                                    except ValidationFailed:
                                        break
                                    if added:
                                        changed = True
                                        already_added[action] += 1
                                    else:
                                        break

                            all_found = True
                            for action, added in already_added.items():
                                if len(adding_queries[action]) > added and action == current_action:
                                    all_found = False

                            if all_found:
                                break
                            elif not changed:
                                print("here it breaks 2")
                                raise ValidationFailed
                        if all_found:
                            break

                    if not all_found:
                        print('main1')
                        raise ValidationFailed

                    all_validation_candidates = AllActionCandidates(validation_trace.action_arity,
                                                                    validation_trace.predicate_arity,
                                                                    validation_trace.predicate_types,
                                                                    validation_problem.objects_types,
                                                                    action_candidate_dict)

                    for t in validation_trace:
                        state = validation_trace.parse_state(validation_trace.get_state_of_index(t))
                        all_validation_candidates.parse_state_validation(state, validation_trace.get_action_name(t),
                                                                         validation_trace.get_action_objects(t), t)

                    if test_object_tuple != cur_object_tuple:
                        raise LookupError
                    if not all_validation_candidates.check_all_unique_queries(all_unique_queries, validation_trace):
                        print("here is breaks")
                        raise ValidationFailed

                    val_state = validation_problem.parse_state_precondition(validation_trace.state_trace[0])

                    full_object_grounding = validation_trace.get_full_grounding()
                    preconditions = preconditions_of_groundings_all[current_action]
                    neg_preconditions = neg_preconditions_of_groundings_all[current_action]

                    validation_trace.set_grounding_preconditions()
                    _, val_masked_pre = validation_problem.parse_state_precondition_test(
                        validation_trace.state_trace[0])
                    for pred in masked_all_positive_preconditions[current_action]:
                        if pred not in val_masked_pre:
                            continue
                        new_precs = validation_trace.get_masked_preconditions(val_masked_pre[pred],
                                                                              full_object_grounding,
                                                                              masked_all_positive_preconditions[
                                                                              current_action][pred],
                                                                              True)
                        if len(new_precs) < len(masked_all_positive_preconditions[current_action][pred]):
                            raise ValidationFailed
                        elif len(new_precs) > 0:
                            print('at some points it works.')


                    for pred in masked_all_negative_preconditions[current_action]:
                        if pred not in val_masked_pre:
                            continue
                        new_precs = validation_trace.get_masked_preconditions(val_masked_pre[pred],
                                                                              full_object_grounding,
                                                                              masked_all_negative_preconditions[
                                                                              current_action][pred],
                                                                              False)
                        if len(new_precs) < len(masked_all_negative_preconditions[current_action][pred]):
                            raise ValidationFailed
                        elif len(new_precs) > 0:
                            print('at some points it works.')

                    possible = True
                    for prec_predicate in preconditions:
                        for prec_pattern in preconditions[prec_predicate]:
                            prec_grounding = tuple([full_object_grounding[pos] for pos in prec_pattern])
                            if prec_predicate not in val_state:
                                possible = False
                                break
                            if prec_grounding not in val_state[prec_predicate]:
                                possible = False
                                break
                        if not possible:
                            break

                    if not possible:
                        continue

                    for prec_predicate in neg_preconditions:
                        for prec_pattern in neg_preconditions[prec_predicate]:
                            prec_grounding = tuple([full_object_grounding[pos] for pos in prec_pattern])
                            if prec_predicate not in val_state:
                                continue
                            if prec_grounding in val_state[prec_predicate]:
                                possible = False
                                break
                        if not possible:
                            break

                    if not possible:
                        continue



                    final_grounding = validation_trace.get_full_grounding()
                    res.add(tuple([current_action, cur_object_tuple]))
                    if tuple([current_action, cur_object_tuple]) not in applicable_action_set:
                        raise OverallValidationFailed

                    new_eff = get_effects_for_grounding(effects, current_action, final_grounding)
                    effects_by_mapping = applicable_actions_with_effects[tuple([current_action, cur_object_tuple])]

                    print('New Effects', new_eff)
                    print('Old Effects', applicable_actions_with_effects[tuple([current_action, cur_object_tuple])])

                    for sign in effects_by_mapping:
                        for predicate in effects_by_mapping[sign]:
                            if effects_by_mapping[sign][predicate] != new_eff[sign][predicate]:
                                raise OverallValidationFailed

                    for sign in new_eff:
                        for predicate in new_eff[sign]:
                            if effects_by_mapping[sign][predicate] != new_eff[sign][predicate]:
                                raise OverallValidationFailed



                    print('VALIDATION WAS SUCCESSFULL!!!', current_action, cur_object_tuple)
                except ValidationFailed:
                    print('\n$$$$$$ All Unique Queries $$$$$$$')
                    for x in all_unique_queries[current_action]:
                        print(x)
                    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
                    print('Validation Failed!!!', current_action, cur_object_tuple)

        print(validation_problem.object_dict)
        if applicable_action_set != res:
            print(res)
            print('To much in output', res - applicable_action_set)
            print('To less in output', applicable_action_set - res)
            raise OverallValidationFailed

        print(seen_actions)
        if {True} == set(seen_actions.values()):
            break
    #else:
    print("\n@@@@@@@@@@@@@@@@@@@@@@\nOverall Validation Correct! \n@@@@@@@@@@@@@@@@@@@@@@\n")


    print('Applicable Actions', res)
    print('object mapping', validation_problem.object_dict)
    return out


def find_additional_arguments_json(trace, add_static, dropped_predicates):

    if dropped_predicates:
        dropped_p = get_dropped_preds(dropped_predicates)
    else:
        dropped_p = set()

    x = JSONThings(trace, add_static)

    num_objects = len(x.all_objects)

    predicate_arities = x.get_feature_arity()

    object_set = x.get_object_set()

    new_Trace = JSONTrace(x, dropped_p)

    initial_arities = copy.deepcopy(new_Trace.action_arity)

    num_initial_args = sum([ar for action, ar in new_Trace.action_arity.items()])
    new_all_things = None

    i = 0
    while True:
        #if i >= 1:   # stop after first iteration
        #    break
        new_Trace.reset_parsed_dict()
        i += 1
        print("This is the {}. iteration".format(i))
        new_all_things = AllActionCandidates(new_Trace.action_arity, predicate_arities, new_Trace.get_predicate_types(),
                                             x.get_object_types(), None)
        for t in new_Trace:
            if new_Trace.action_name_list[t] == '':
                continue
            #action_name = new_Trace.get_action_name(t)
            #if new_Trace.action_arity[action_name] >= 10:
            #    continue  # skip this action, don't add more arguments

            new_parsed_state = new_Trace.parse_state(new_Trace.get_state_of_index(t))
            new_all_things.parse_state(new_parsed_state, new_Trace.get_action_name(t), new_Trace.get_action_objects(t),
                                       t)

        was_there_somehting_added = new_all_things.add_arguments(new_Trace)
        
        #if was_there_somehting_added:
        #    added_any = True

        #combi_added = False
        combi_added = new_all_things.check_combis(new_Trace)

        if not was_there_somehting_added and not combi_added:
            # if not was_there_somehting_added:
            break

    print(new_Trace.action_arity)

    effects = new_Trace.get_effect_argument_positions()
    print_effects(effects)

    print(effects)

    new_Trace.print_query_output()

    # new_all_things.set_unique_patterns()
    all_unique_queries = new_all_things.get_unique_queries()
    adding_queries = new_Trace.get_queries()
    effect_mapping = new_Trace.get_effect_mapping()

    print('\n-----------------------------\n')

    print('%%%%%%   Adding Querries   %%%%')
    for a, q in adding_queries.items():
        for qq in q:
            print(a, qq)

    print('\n %%%%%% Unique quries %%%%%%')
    for a, q in all_unique_queries.items():
        for qq in q:
            print(a, qq)

    print('\n-----------------------------\n')
    
    #new_Trace.set_grounding_preconditions()

    new_Trace.set_grounding_preconditions()

    new_Trace.get_pddl_action_model()

    new_Trace.print_query_output()
   
    
def pddl_output(trace, effects, preconditions):

    print('test')


def all_tests_none():

    all_test_dict = dict()

    all_test_dict["bw3"] = ['pddl_files/blocks_3/blocks_world.pddl',
                        'pddl_files/blocks_3/blocks_world_5.pddl',
                        500,
                        None, None]
    all_test_dict["bw4"] = ['pddl_files/blocks_4/blocks_4ops.pddl',
                        'pddl_files/blocks_4/blocks_4ops_5.pddl',
                        500,
                        None, None]
    all_test_dict["ferry"] = ['pddl_files/ferry/ferry.pddl',
                        'pddl_files/ferry/ferry-4-4.pddl',
                        500,
                        None, None]
    all_test_dict["npuzzle"] = ['pddl_files/npuzzle/npuzzle.pddl',
                        'pddl_files/npuzzle/npuzzle-3-3.pddl',
                        500,
                        None, None]
    all_test_dict["satellite"] = ['pddl_files/satellite/satellite.pddl',
                        'pddl_files/satellite/satellite_9.pddl',
                        10000,
                        None, None]
    all_test_dict["delivery"] = ['pddl_files/delivery/delivery.pddl',
                                  'pddl_files/delivery/delivery-3-3-4-2.pddl',
                                  1000,
                                  None, None]
    all_test_dict["driverlog"] = ['pddl_files/driverlog/driverlog.pddl',
                                  'pddl_files/driverlog/driverlog-1.pddl',
                                  1000,
                                  None, None]
    all_test_dict["grid"] = ['pddl_files/grid/grid.pddl',
                                  'pddl_files/grid/grid-4.pddl',
                                  1000,
                                  None, None]
    all_test_dict["gridlock"] = ['pddl_files/grid/lock_grid.pddl',
                                  'pddl_files/grid/grid-4.pddl',
                                  1000,
                                  None, None]
    all_test_dict["gripper"] = ['pddl_files/gripper/gripper.pddl',
                                  'pddl_files/gripper/gripper_6_2.pddl',
                                  1000,
                                  None, None]
    all_test_dict["hanoi"] = ['pddl_files/hanoi/hanoi.pddl',
                                  'pddl_files/hanoi/hanoi-3-2.pddl',
                                  1000,
                                  None, None]
    all_test_dict["logistics"] = ['pddl_files/logistics/logistics.pddl',
                              'pddl_files/logistics/logistics_3_3_3_3_3_3.pddl',
                              1000,
                              None, None]
    all_test_dict["miconic"] = ['pddl_files/miconic/miconic.pddl',
                              'pddl_files/miconic/miconic_5_4.pddl',
                              1000,
                              None, None]
    all_test_dict["sokoban"] = ['pddl_files/sokoban/sokoban.pddl',
                              'pddl_files/sokoban/sokoban-4-4_3.pddl',
                              1000,
                              None, None]
    all_test_dict["sokoban-pull"] = ['pddl_files/sokoban/pull_sokoban.pddl',
                              'pddl_files/sokoban/sokoban-4-4_3.pddl',
                              1000,
                              None, None]


def all_tests(reduce_args):
    all_test_dict = dict()

    '''
    all_test_dict["bw3-1"] = ['pddl_files/blocks_3/blocks_world.pddl',
                              'pddl_files/blocks_3/blocks_world_5.pddl',
                              100,
                              'dropped_args/blocks3_1.txt',
                              None,
                              'pddl_files/blocks_3/blocks_world_6.pddl',
                                 None]
    all_test_dict["bw3-2"] = ['pddl_files/blocks_3/blocks_world.pddl',
                              'pddl_files/blocks_3/blocks_world_5.pddl',
                              100,
                              'dropped_args/blocks3_2.txt',
                              None,
                              'pddl_files/blocks_3/blocks_world_6.pddl',
                                 None ]
    all_test_dict["bw4-1"] = ['pddl_files/blocks_4/blocks_4ops.pddl',
                              'pddl_files/blocks_4/blocks_4ops_5.pddl',
                              100,
                              'dropped_args/blocks4_1.txt',
                              None,
                              'pddl_files/blocks_4/blocks_4ops_6.pddl',
                                 None]
    all_test_dict["bw4-2"] = ['pddl_files/blocks_4/blocks_4ops.pddl',
                              'pddl_files/blocks_4/blocks_4ops_5.pddl',
                              500,
                              'dropped_args/blocks4_2.txt',
                              None,
                              'pddl_files/blocks_4/blocks_4ops_6.pddl',
                                 None]
    all_test_dict["ferry"] = ['pddl_files/ferry/ferry.pddl',
                              'pddl_files/ferry/ferry-4-4.pddl',
                              100,
                              'dropped_args/ferry.txt',
                              None,
                              'pddl_files/ferry/ferry-5-5.pddl',
                                 None]
    # all_test_dict["npuzzle-3-3"] = ['pddl_files/npuzzle/npuzzle.pddl',
    #                            'pddl_files/npuzzle/npuzzle-3-3.pddl',
    #                            1000,
    #                            'dropped_args/npuzzle_1.txt', None]
    # all_test_dict["npuzzle-4-4"] = ['pddl_files/npuzzle/npuzzle.pddl',
    #                            'pddl_files/npuzzle/npuzzle-4-4.pddl',
    #                            500,
    #                            'dropped_args/npuzzle_1.txt', None]
    all_test_dict["npuzzle-5-5"] = ['pddl_files/npuzzle/npuzzle.pddl',
                                    'pddl_files/npuzzle/npuzzle-5-5.pddl',
                                    500,
                                    'dropped_args/npuzzle_1.txt', None, None,
                                 None]
    all_test_dict["cell-npuzzle"] = ['pddl_files/npuzzle/c-npuzzle.pddl',
                                     'pddl_files/npuzzle/c-5-5-npuzzle.pddl',
                                     500,
                                     'dropped_args/cpuzzle.txt', None, None,
                                 None]
    # all_test_dict["satellite"] = ['pddl_files/satellite/satellite.pddl',
    #                              'pddl_files/satellite/satellite_9.pddl',
    #                              4000,
    #                              'dropped_args/satellite_1.txt', None]
    '''

    '''
    all_test_dict["delivery"] = ['pddl_files/delivery/delivery.pddl',
                                 'pddl_files/delivery/delivery-3-3-4-3.pddl',
                                 1000,
                                 'dropped_args/delivery.txt',
                                 None,
                                 'pddl_files/delivery/delivery-4-4-4-4.pddl',
                                 None]
    '''
    all_test_dict["driverlog-huge"] = ['pddl_files/driverlog/driverlog.pddl',
                                       'pddl_files/driverlog/driverlog-mid.pddl',
                                       3000,
                                       'dropped_args/driverlog.txt',
                                       None,
                                       'pddl_files/driverlog/driverlog-huge.pddl',
                                 None]
    '''
    all_test_dict["grid-huge"] = ['pddl_files/grid/grid.pddl',
                                  'pddl_files/grid/grid-huge.pddl',
                                  10000,
                                  'dropped_args/grid.txt',
                                  None,
                                  'pddl_files/grid/grid-huge-2.pddl',
                                 None]
    all_test_dict["gridlock"] = ['pddl_files/grid/lock_grid.pddl',
                                 'pddl_files/grid/grid-huge.pddl',
                                 10000,
                                 'dropped_args/grid_lock.txt',
                                 None,
                                 'pddl_files/grid/grid-huge-2.pddl',
                                 None]
    '''

    '''
    all_test_dict["gripper-1"] = ['pddl_files/gripper/gripper.pddl',
                                  'pddl_files/gripper/gripper_6_2.pddl',
                                  500,
                                  'dropped_args/gripper_1.txt',
                                  None,
                                  'pddl_files/gripper/gripper_8_2.pddl',
                                 None]
    all_test_dict["gripper-2"] = ['pddl_files/gripper/gripper.pddl',
                                  'pddl_files/gripper/gripper_6_2.pddl',
                                  500,
                                  'dropped_args/gripper_2.txt',
                                  None,
                                  'pddl_files/gripper/gripper_8_2.pddl',
                                 None]

    all_test_dict["hanoi-1"] = ['pddl_files/hanoi/hanoi.pddl',
                                'pddl_files/hanoi/hanoi-3-5.pddl',
                                200,
                                'dropped_args/hanoi_1.txt',
                                None,
                                'pddl_files/hanoi/hanoi-3-7.pddl',
                                 None]
    all_test_dict["hanoi-2"] = ['pddl_files/hanoi/hanoi.pddl',
                                'pddl_files/hanoi/hanoi-3-3.pddl',
                                200,
                                'dropped_args/hanoi_2.txt',
                                None,
                                'pddl_files/hanoi/hanoi-3-7.pddl',
                                 None]
    
    # all_test_dict["logistics-1"] = ['pddl_files/logistics/logistics.pddl',
    #                              'pddl_files/logistics/logistics_3_3_3_3_3_3.pddl',
    #                              1000,
    #                              'dropped_args/logistics.txt', None, None]
    all_test_dict["logistics-1"] = ['pddl_files/logistics/logistics.pddl',
                                    'pddl_files/logistics/logistics-1.pddl',
                                    1000,
                                    'dropped_args/logistics.txt',
                                    None,
                                    'pddl_files/logistics/logistics_3_3_3_3_3_3.pddl',
                                 None]
    '''
    all_test_dict["logistics-2"] = ['pddl_files/logistics/logistics.pddl',
                                    'pddl_files/logistics/logistics-t1.pddl',
                                    1000,
                                    'dropped_args/logistics.txt',
                                    None,
                                    'pddl_files/logistics/logistics-t2.pddl',
                                 None]
    '''
    all_test_dict["miconic"] = ['pddl_files/miconic/miconic.pddl',
                                'pddl_files/miconic/miconic_5_4.pddl',
                               200,
                                'dropped_args/miconic.txt',
                                None,
                                'pddl_files/miconic/miconic_6_6.pddl',
                                 None]
    '''
    all_test_dict["sokoban-1"] = ['pddl_files/sokoban/sokoban.pddl',
                                  'pddl_files/sokoban/sokoban-5-5_3.pddl',
                                  400,
                                  'dropped_args/sokoban_1.txt',
                                  None,
                                  'pddl_files/sokoban/sokoban-6-5_4.pddl',
                                 None]
    '''
    all_test_dict["sokoban-2"] = ['pddl_files/sokoban/sokoban.pddl',
                                  'pddl_files/sokoban/sokoban-5-5_3.pddl',
                                  400,
                                  'dropped_args/sokoban_2.txt',
                                  None,
                                  'pddl_files/sokoban/sokoban-5-5_3.pddl',
                                 None]
    '''
    all_test_dict["sokoban-pull-1"] = ['pddl_files/sokoban/pull_sokoban.pddl',
                                       'pddl_files/sokoban/sokoban-5-5_3.pddl',
                                       400,
                                       'dropped_args/pull_sokoban-1.txt',
                                       None,
                                       'pddl_files/sokoban/sokoban-6-5_4.pddl',
                                 None]
    '''
    all_test_dict["sokoban-pull-2"] = ['pddl_files/sokoban/pull_sokoban.pddl',
                                       'pddl_files/sokoban/sokoban-5-5_3.pddl',
                                       400,
                                       'dropped_args/pull_sokoban-2.txt',
                                       None,
                                       'pddl_files/sokoban/sokoban-5-5_3.pddl',
                                 None]
    '''
    result_dict = dict()
    
    for key, test in all_test_dict.items():

        print(key)

        # Create a StringIO buffer to capture output
        buffer = io.StringIO()

        start = time.time()
        validated = True
        # Redirect stdout to the buffer
        with redirect_stdout(buffer):
            if not reduce_args:
                try:
                    result_dict[key] = find_additional_arguments(*test)
                except (ValidationFailed, OverallValidationFailed) as e:
                    validated = False
            else:
                try:
                    result_dict[key] = reduce_strips(test[0], test[1], test[2], test[4])
                except  (ValidationFailed, OverallValidationFailed) as e:
                    validated = False
        if not validated:
            print('VALIDATION FAILED!!!')

        print(time.time() - start)

        if reduce_args:
            if validated:
                print('Is all patterns smaller: ', result_dict[key][0])
                print("Size of action_schema only precondition:", result_dict[key][1])
                print("Size of action_schema all pattern:", result_dict[key][2])
            else:
                print("HERE HAPPENED SOMETHING WEIRD")

        # Get all printed content from the buffer
        output = buffer.getvalue()

        # Write the captured output to a file
        if reduce_args:
            with open("output/reduce_{}.txt".format(key), "w") as f:
                f.write(output)
        else:
            with open("output/{}.txt".format(key), "w") as f:
                f.write(output)

    if not reduce_args:
        result_table = "\\documentclass{article} \n"
        result_table += "\\RequirePackage{geometry}\\geometry{left=0.25in,right=0.25in,top=1in,bottom=1in} \n"
        result_table += '\\begin{document} \n'
        result_table += "\\begin{table}[] \n\\begin{tabular}{lllllllll} \n"
        result_table += 'Domain & Domain Args & Input Args & Output Args & Missing Args & Additional Args & time & Size & Num Objects'

        for key, value in result_dict.items():

            result_table += ' \\\\ \n'

            result_row = copy.deepcopy(key)

            for val in value:
                result_row += ' & ' + str(val)

            result_table += result_row

        result_table += '\n\\end{tabular} \n\\end{table} \n'
        result_table += '\\end{document} \n'

        print(result_table)

        with open('latex/tables_test_1.txt', 'w') as file:
            file.write(result_table)

    return None


def test_all_dropped_predicates():

    all_dropped_test = dict()

    all_dropped_test["ferry"] = ['pddl_files/ferry/ferry.pddl',
                              'pddl_files/ferry/ferry-4-4.pddl',
                              300,
                              'dropped_args/ferry.txt',
                              'dropped_predicates/ferry.txt',
                              'pddl_files/ferry/ferry-5-5.pddl',
                                 None]

    all_dropped_test["bw3"] = ['pddl_files/blocks_3/blocks_world.pddl',
                              'pddl_files/blocks_3/blocks_world_5.pddl',
                              200,
                              'dropped_args/blocks3_1.txt',
                              'dropped_predicates/blocks.txt',
                              'pddl_files/blocks_3/blocks_world_6.pddl',
                                 None]

    all_dropped_test["cpuzzle-5-5"] = ['pddl_files/npuzzle/c-npuzzle.pddl',
                                     'pddl_files/npuzzle/c-5-5-npuzzle.pddl',
                                    500,
                                    'dropped_args/cpuzzle.txt',
                                       'dropped_predicates/cpuzzle.txt',
                                       None,
                                 None]

    all_dropped_test["miconic"] = ['pddl_files/miconic/miconic.pddl',
                                'pddl_files/miconic/miconic_5_4.pddl',
                                200,
                                'dropped_args/miconic.txt',
                                'dropped_predicates/miconic.txt',
                                'pddl_files/miconic/miconic_6_6.pddl',
                                 None]

    result_dict = dict()

    for key, test in all_dropped_test.items():

        print(key)
        # Create a StringIO buffer to capture output
        buffer = io.StringIO()

        start = time.time()
        # Redirect stdout to the buffer
        failed = False
        with redirect_stdout(buffer):
            try:
                result_dict[key] = find_additional_arguments(*test)
            except ValidationFailed:
                failed = True

        if failed:
            print(print('VALIDATION FAILED!!!'))

        print(time.time() - start)

        # Get all printed content from the buffer
        output = buffer.getvalue()

        # Write the captured output to a file

        with open("output/dropped_predicate{}.txt".format(key), "w") as f:
            f.write(output)


    result_table = "\\documentclass{article} \n"
    result_table += "\\RequirePackage{geometry}\\geometry{left=0.25in,right=0.25in,top=1in,bottom=1in} \n"
    result_table += '\\begin{document} \n'
    result_table += "\\begin{table}[] \n\\begin{tabular}{lllllllll} \n"
    result_table += 'Domain & Domain Args & Input Args & Output Args & Missing Args & Additional Args & time & Size & Num Objects'

    for key, value in result_dict.items():

        result_table += ' \\\\ \n'

        result_row = copy.deepcopy(key)

        for val in value:
            result_row += ' & ' + str(val)

        result_table += result_row

    result_table += '\n\\end{tabular} \n\\end{table} \n'
    result_table += '\\end{document} \n'

    print(result_table)

    with open('latex/table_dropped_predicates.tex', 'w') as file:
        file.write(result_table)

    return None

def drop_test(domain, instance, size, dropped_args):

    x = mimir_thing(domain, instance, set(), dropped_args)

    predicates = [pred for (pred,_) in x.get_feature_arity() if pred not in x.unary_static_predicates]
    droppable_predicates = set()

    for x in predicates:
        print(x)
        buffer = io.StringIO()
        # Redirect stdout to the buffer
        failed = False
        with redirect_stdout(buffer):
            try:
                res = find_additional_arguments(domain, instance, size, dropped_args, {x}, None, None)
            except (ValidationFailed, OverallValidationFailed) as e:
                failed = True
                continue
        if not failed:
        #output = buffer.getvalue()
            if res[3] == 0:
                droppable_predicates.add(x)
                print(x)

    print("Predicates that can be dropped:", droppable_predicates)

    current_predicates = [{x} for x in droppable_predicates]
    not_possible_combis = set()

    while True:

        next_predicates = []
        seen_sets = set()

        for x in current_predicates:
            for y in droppable_predicates:

                if y in x:
                    continue

                new_set = copy.deepcopy(x)
                new_set.add(y)

                if frozenset(new_set) in seen_sets:
                    continue

                not_possible = False

                for negative_set in not_possible_combis:
                    if negative_set.issubset(new_set):
                        not_possible = True
                        break

                if not_possible:
                    continue

                seen_sets.add(frozenset(new_set))

                buffer = io.StringIO()
                # Redirect stdout to the buffer
                with redirect_stdout(buffer):
                    try:
                        res = find_additional_arguments(domain, instance, size, dropped_args, new_set, None, None)
                    except ValidationFailed:
                        continue
                output = buffer.getvalue()

                if res[3] == 0:
                    next_predicates.append(new_set)
                    print("the combination of the predicates can be dropped:", new_set)
                else:
                    not_possible_combis.add(frozenset(new_set))

        if len(next_predicates) == 0:
            print("Largest possible combinations",current_predicates)
            break
        current_predicates = next_predicates


if __name__ == '__main__':

    print('test')

    args = get_this_args()

    #if args.json:
    #    print('test')
    #    find_additional_arguments_json("test", None)

    if args.single_test:

        buffer = io.StringIO()
        # Redirect stdout to the buffer
        failed = False
        with redirect_stdout(buffer):
            try:
                 res = find_additional_arguments(args.domain, args.instance, args.size, None, args.dropped_predicates,
                                          args.validation_instance, args.validation_size)
            except (ValidationFailed, OverallValidationFailed) as e:
                failed = True

        output = buffer.getvalue()

        if not os.path.exists('output/extended_output/'):
            os.makedirs('output/extended_output/')

        # print(output)
        with open('output/extended_output/{}_{}.txt'.format(args.name, args.run), 'w') as file:
            file.write(output)

        if not failed:
            print(res)
            with open('output/{}_{}.txt'.format(args.name, args.run), 'w') as file:
                file.write(str(res))
    elif args.test_drop:

        buffer = io.StringIO()
        # Redirect stdout to the buffer
        failed = False
        with redirect_stdout(buffer):
            try:
                res = find_additional_arguments(args.domain, args.instance, args.size, None, args.dropped_predicates,
                                                args.validation_instance, args.validation_size)
            except (ValidationFailed, OverallValidationFailed) as e:
                failed = True

        output = buffer.getvalue()

        if not os.path.exists('output/extended_output/'):
            os.makedirs('output/extended_output/')

        # print(output)
        with open('output/extended_output/dropped_{}_{}.txt'.format(args.name, args.run), 'w') as file:
            file.write(output)

        if not failed:
            print(res)
            with open('output/dropped_{}_{}.txt'.format(args.name, args.run), 'w') as file:
                file.write(str(res))

    elif args.testall:
        all_tests(args.reduce_strips)
    elif args.test_all_dropped:
        test_all_dropped_predicates()
    elif args.drop_test:
        drop_test(args.domain, args.instance, args.size, args.dropped_args)
    elif args.reduce_strips:
        reduce_strips(args.domain, args.instance, args.size, args.dropped_args)
    elif args.json:
        find_additional_arguments_json(args.json, args.json_add_static, None)
    else:
        find_additional_arguments(args.domain, args.instance, args.size, args.dropped_args, args.dropped_predicates,
                                  args.validation_instance, None)
