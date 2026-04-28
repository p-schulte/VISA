(define (domain hanoi)
    (:requirements :strips)
    (:predicates
      (clear ?x)
      (on ?x ?y)
      (smaller ?x ?y)
      ; (move ?disc ?to) OLD ACTION
      (handempty ?x)
      (handfull ?x)
      (holding ?x)
      (stack ?x ?y)
      (unstack ?x)
    )


    ; (:actions stack unstack)
    (:action stack
        :parameters (?x ?y ?robot)
        :precondition (and
            (stack ?x ?y)
            (holding ?x) 
            (clear ?y)
            (smaller ?x ?y)
            (handfull ?robot)
        )
        :effect (and 
            (not (holding ?x))
            (not (clear ?y))
            (clear ?x)
            (handempty ?robot)
            (not (handfull ?robot))
            (on ?x ?y)
        )
    )

    (:action unstack
        :parameters (?x ?y ?robot)
        :precondition (and
            (unstack ?x)
            (on ?x ?y)
            (clear ?x)
            (handempty ?robot)
        )
        :effect (and 
            (holding ?x)
            (clear ?y)
            (not (clear ?x))
            (not (handempty ?robot))
            (handfull ?robot)
            (not (on ?x ?y))
        )
    )
)


  ; (:actions move)

    ; (:action move    OLD ACTION DEFINITION
    ;   :parameters (?disc ?from ?to)
    ;   :precondition (and 
    ;     (move ?disc ?to) 
    ;     (smaller ?disc ?to) 
    ;     (on ?disc ?from)
    ;     (clear ?disc) 
    ;     (clear ?to)
    ;   )
    ;   :effect  (and 
    ;     (clear ?from) 
    ;     (on ?disc ?to) 
    ;     (not (on ?disc ?from))
    ;     (not (clear ?to))
    ;   )
    ; )

