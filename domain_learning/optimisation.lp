%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Optimization ordered by priority 

%#minimize {1@0, not_possible_object_combinations(Index,Objects): statics(Index), not_possible_object_combinations(Index,Objects), obj_tuple(Arity,Objects)}.

% prefer models with the lowest amount of statics
#minimize {(Arity+1)**(Arity)@2, P: pattern(I,P), pattern(P,Actin,Arity,Objects)}.

% prefer models with minimum sum of static cardinality
%#minimize { (N)**(N+1)@2, A : static(A, N) }.

% prefer models with less patterns
%#minimize { 1@1, A: pattern(A,Index) }.

#minimize {N@1:number_statics(N)}.
