(define (domain gripper)
	(:requirements :strips :typing)
	(:types room ball gripper)
   (:predicates (at-robby ?r - room)
        (isroom-a ?r - room)
        (isroom-b ?r - room)
		(at ?b - ball ?r - room)
		(free ?g - gripper)
		(carry ?b - ball ?g - gripper)
        (different ?r1 - room ?r2 - room)
        (above ?r1 - room ?r2 - room)
        (move ?from - room ?to - room)
        (up ?from - room ?to - room)
        (down ?from - room ?to - room)
        (pick ?ball - ball ?room - room ?gripper - gripper)
        (drop ?ball - ball ?room - room ?gripper - gripper)
        
    )

    ; (:actions move up down pick drop)

   (:action move
       :parameters  (?from - room ?to - room)
       :precondition (and (move ?from ?to) (at-robby ?from) (different ?from ?to))
       :effect (and  (at-robby ?to)
		     (not (at-robby ?from))))

   (:action up
       :parameters  (?from - room ?to - room)
       :precondition (and (up ?from ?to) (at-robby ?from) (above ?to ?from))
       :effect (and  (at-robby ?to)
		     (not (at-robby ?from))))

   (:action down
       :parameters  (?from - room ?to - room)
       :precondition (and (down ?from ?to) (at-robby ?from) (above ?from ?to))
       :effect (and  (at-robby ?to)
		     (not (at-robby ?from))))


   (:action pick
       :parameters (?ball - ball ?room - room ?gripper - gripper)
       :precondition  (and (pick ?ball  ?room  ?gripper ) (at ?ball ?room) (at-robby ?room) (free ?gripper))
       :effect (and (carry ?ball ?gripper)
		    (not (at ?ball ?room)) 
		    (not (free ?gripper))))


   (:action drop
       :parameters  (?ball - ball ?room - room ?gripper - gripper)
       :precondition  (and (drop ?ball  ?room  ?gripper ) (carry ?ball ?gripper) (at-robby ?room))
       :effect (and (at ?ball ?room)
		    (free ?gripper)
		    (not (carry ?ball ?gripper)))))