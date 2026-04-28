; Robots can only move if they have an ecell installed
(define (domain energy_robots)
  (:requirements :negative-preconditions :equality :typing)
  (:types         location locatable - object
		robot ecell - locatable
  )
  (:predicates 
		(at ?obj - locatable ?loc - location)
		(used ?c - ecell ?r - robot)
		(holding ?o - ecell ?r - robot)
		(handfree ?r - robot)
		(charged ?r - robot)
  )


  (:action CHARGE
    :parameters
    (?c - ecell
      ?r - robot
      ?loc - location)
    :precondition
    (and (at ?r ?loc) (at ?c ?loc) (not (charged ?r)))
    :effect
    (and (not (at ?c ?loc)) (used ?c ?r) (charged ?r)))

  (:action UNCHARGE
    :parameters
    (?c - ecell
      ?r - robot
      ?loc - location)
    :precondition
    (and (at ?r ?loc) (used ?c ?r))
    :effect
    (and (at ?c ?loc) (not (charged ?r)) (not (used ?c ?r))))

  (:action MOVE
    :parameters
    ( ?r - robot
      ?loc - location
      ?tar - location)
    :precondition
    (and (at ?r ?loc) (charged ?r) (not (= ?loc ?tar)))
    :effect
    (and (at ?r ?tar) (not (at ?r ?loc))))

  (:action PICK
    :parameters
    (?c - ecell
      ?r - robot
      ?loc - location)
    :precondition
    (and (at ?r ?loc) (at ?c ?loc) (handfree ?r))
    :effect
    (and (not (at ?c ?loc)) (holding ?c ?r) (not (handfree ?r))))

  (:action PLACE
    :parameters
    (?c - ecell
      ?r - robot
      ?loc - location)
    :precondition
    (and (at ?r ?loc) (holding ?c ?r))
    :effect
    (and (at ?c ?loc) (handfree ?r) (not (holding ?c ?r))))

)