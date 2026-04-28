% input to this problem are possible actions and not possible actions together
#const max_static=10.
#const max_arity=3.
#const num_patterm=100.
#const max_types=100.

% how many statics 
{number_statics(1..max_static)} = 1.
statics(1..N) :- number_statics(N).

% create statics with arity
{static(I,1..max_arity)} = 1 :- statics(I).   

%%% pattern
%%% pattern(index,action,arity,pattern).

% choose subset of preconditions 
1 {pattern(I,Index):pattern(Index,_,Arity,_)} :- static(I,Arity).

% choose subset of objets the static holds or does not hold 
1 {not_possible_object_combinations(Index,Objects):obj_tuple(Arity,Objects)} :- static(Index,Arity).

mapping((O0,O1,O2),(0),O0) :-     pos_action(_,(O0,O1,O2)), O0 > -1.
mapping((O0,O1,O2),(1),O1) :-     pos_action(_,(O0,O1,O2)), O1 > -1.
mapping((O0,O1,O2),(2),O2) :-     pos_action(_,(O0,O1,O2)), O2 > -1.
mapping((O0,O1,O2),(0,0),(O0,O0)) :-pos_action(_,(O0,O1,O2)), O0 > -1.
mapping((O0,O1,O2),(0,1),(O0,O1)) :-pos_action(_,(O0,O1,O2)), O1 > -1.
mapping((O0,O1,O2),(0,2),(O0,O2)) :-pos_action(_,(O0,O1,O2)), O2 > -1.
mapping((O0,O1,O2),(1,0),(O1,O0)) :-pos_action(_,(O0,O1,O2)), O1 > -1.
mapping((O0,O1,O2),(1,1),(O1,O1)) :-pos_action(_,(O0,O1,O2)), O1 > -1.
mapping((O0,O1,O2),(1,2),(O1,O2)) :-pos_action(_,(O0,O1,O2)), O2 > -1.
mapping((O0,O1,O2),(2,0),(O2,O0)) :-pos_action(_,(O0,O1,O2)), O2 > -1.
mapping((O0,O1,O2),(2,1),(O2,O1)) :-pos_action(_,(O0,O1,O2)), O2 > -1.
mapping((O0,O1,O2),(2,2),(O2,O2)) :-pos_action(_,(O0,O1,O2)), O2 > -1.
mapping((O0,O1,O2),(0),O0) :-     neg_action(_,(O0,O1,O2)), O0 > -1.
mapping((O0,O1,O2),(1),O1) :-     neg_action(_,(O0,O1,O2)), O1 > -1.
mapping((O0,O1,O2),(2),O2) :-     neg_action(_,(O0,O1,O2)), O2 > -1.
mapping((O0,O1,O2),(0,0),(O0,O0)) :-neg_action(_,(O0,O1,O2)), O0 > -1.
mapping((O0,O1,O2),(0,1),(O0,O1)) :-neg_action(_,(O0,O1,O2)), O1 > -1.
mapping((O0,O1,O2),(0,2),(O0,O2)) :-neg_action(_,(O0,O1,O2)), O2 > -1.
mapping((O0,O1,O2),(1,0),(O1,O0)) :-neg_action(_,(O0,O1,O2)), O1 > -1.
mapping((O0,O1,O2),(1,1),(O1,O1)) :-neg_action(_,(O0,O1,O2)), O1 > -1.
mapping((O0,O1,O2),(1,2),(O1,O2)) :-neg_action(_,(O0,O1,O2)), O2 > -1.
mapping((O0,O1,O2),(2,0),(O2,O0)) :-neg_action(_,(O0,O1,O2)), O2 > -1.
mapping((O0,O1,O2),(2,1),(O2,O1)) :-neg_action(_,(O0,O1,O2)), O2 > -1.
mapping((O0,O1,O2),(2,2),(O2,O2)) :-neg_action(_,(O0,O1,O2)), O2 > -1.

% each possible action can not be cut of 
% pos_action(action,objects)
not_applicable(Action,Objects) :- pos_action(Action, Objects), statics(I), pattern(I,Index), pattern(Index,Action,_,Pattern), mapping(Objects, Pattern, Combination), not_possible_object_combinations(I,Combination).
not_applicable(Action,Objects) :- neg_action(Action, Objects), statics(I), pattern(I,Index), pattern(Index,Action,_,Pattern), mapping(Objects, Pattern, Combination), not_possible_object_combinations(I,Combination).


:- not_applicable(Action,Objects), pos_action(Action,Objects).
:- not not_applicable(Action,Objects), neg_action(Action,Objects).


#show static/2.
%#show not_applicable/2.
%#show not_possible_object_combinations/2.
#show pattern/2.
%#show not_possible_object_combinations/2.