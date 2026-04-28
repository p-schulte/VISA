;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 4 op-blocks world
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain ball_hiding)
    (:requirements :strips :typing)
    (:types block ball robot)
    (:predicates 
        (on ?x - block ?y - block)
        (on ?x - ball ?y - block)
        (ontable ?x - block)
        (ontable ?x - ball)
        (clear ?x - block)
        (handempty ?x - robot)
        (handfull ?x - robot)
        (holding ?x - block)
        (holding ?x - ball)
        (block_full ?x - block)
        (block_empty ?x - block)
        (in ?x - ball ?y - block)

        (pickup_block ?x - block)
        (pickup_ball ?x - ball)
        (putdown_block ?x - block)
        (putdown_ball ?x - ball)
        (stack_block ?x - block ?y - block)
        (stack_ball ?x - ball ?y - block)
        (unstack_block ?x - block)
        (unstack_ball ?x - ball)
        (hide_ball ?x - ball ?y - block)
        (unhide_ball ?x - ball ?y - block)
    )

    ; (:actions pickup_block pickup_ball putdown_block putdown_ball stack_block stack_ball unstack_block unstack_ball hide_ball unhide_ball)

    (:action pickup_block
        :parameters (?x - block ?robot - robot)
        :precondition (and
            (pickup_block ?x) 
            (clear ?x) 
            (block_empty ?x)
            (ontable ?x) 
            (handempty ?robot)
        )
        :effect (and
            (not (ontable ?x))
            (not (clear ?x))
            (not (handempty ?robot))
            (handfull ?robot)
            (holding ?x)
        )
    )

    (:action pickup_ball
        :parameters (?x - ball ?robot - robot)
        :precondition (and
            (pickup_ball ?x) 
            (ontable ?x) 
            (handempty ?robot)
        )
        :effect (and
            (not (ontable ?x))
            (not (handempty ?robot))
            (handfull ?robot)
            (holding ?x)
        )
    )






    (:action putdown_block
        :parameters (?x - block ?robot - robot)
        :precondition (and 
            (putdown_block ?x)
            (holding ?x)
            (handfull ?robot)
        )
        :effect (and 
            (not (holding ?x))
            (clear ?x)
            (handempty ?robot)
            (not (handfull ?robot))
            (ontable ?x))
    )

    (:action putdown_ball
        :parameters (?x - ball ?robot - robot)
        :precondition (and 
            (putdown_ball ?x)
            (holding ?x)
            (handfull ?robot)
        )
        :effect (and 
            (not (holding ?x))
            (handempty ?robot)
            (not (handfull ?robot))
            (ontable ?x))
    )






    (:action stack_block
        :parameters (?x - block ?y - block ?robot - robot)
        :precondition (and
            (stack_block ?x ?y)
            (holding ?x) 
            (clear ?y)
            (block_empty ?y)
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

    (:action stack_ball
        :parameters (?x - ball ?y - block ?robot - robot)
        :precondition (and
            (stack_ball ?x ?y)
            (holding ?x) 
            (handfull ?robot)
            (clear ?y)
            (block_empty ?y)
        )
        :effect (and 
            (not (holding ?x))
            (not (clear ?y))
            (handempty ?robot)
            (not (handfull ?robot))
            (on ?x ?y)
        )
    )







    (:action unstack_block
        :parameters (?x - block ?y - block ?robot - robot)
        :precondition (and
            (unstack_block ?x)
            (on ?x ?y)
            (clear ?x)
            (block_empty ?x)
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

    (:action unstack_ball
        :parameters (?x - ball ?y - block ?robot - robot)
        :precondition (and
            (unstack_ball ?x)
            (on ?x ?y)
            (handempty ?robot)
        )
        :effect (and 
            (holding ?x)
            (clear ?y)
            (not (handempty ?robot))
            (handfull ?robot)
            (not (on ?x ?y))
        )
    )

    




    (:action hide_ball
        :parameters (?x - ball ?y - block ?robot - robot)
        :precondition (and
            (hide_ball ?x ?y)
            (block_empty ?y)
            (on ?x ?y)
            (handempty ?robot)
        )
        :effect (and 
            (block_full ?y)
            (not (block_empty ?y))
            (in ?x ?y)
            (not (on ?x ?y))
        )
    )

    (:action unhide_ball
        :parameters (?x - ball ?y - block ?robot - robot)
        :precondition (and
            (unhide_ball ?x ?y)
            (in ?x ?y)
            (block_full ?y)
            (handempty ?robot)
        )
        :effect (and 
            (not (block_full ?y))
            (block_empty ?y)
            (not (in ?x ?y))
            (on ?x ?y)
        )
    )
)
