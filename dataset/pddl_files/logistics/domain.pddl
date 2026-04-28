(define (domain logistics)
  (:requirements :strips :typing)
  (:types
    movable location city - object
    obj transport - movable
    truck airplane - transport
    airport - location
  )
  (:predicates
    (at ?obj - movable ?loc - location)
    (in ?obj1 - obj ?obj2 - transport)
    (in-city ?loc - location ?city - city)
    (loadtruck ?obj - obj ?truck - truck ?loc - location)
    (loadairplane ?obj - obj ?airplane - airplane ?loc - airport)
    (unloadtruck ?obj - obj ?truck - truck ?loc - location)
    (unloadairplane ?obj - obj ?airplane - airplane ?loc - airport)
    (drivetruck ?truck - truck ?locfrom - location ?locto - location ?city - city)
    (flyairplane ?airplane - airplane ?locfrom - airport ?locto - airport)
    (different ?l1 - location ?l2 - location)   ;; new predicate for inequality
  )

  
  ; (:actions loadtruck loadairplane unloadtruck unloadairplane drivetruck flyairplane)

  (:action load-truck
    :parameters (?obj - obj ?truck - truck ?loc - location)
    :precondition (and (loadtruck ?obj ?truck ?loc) (at ?truck ?loc) (at ?obj ?loc))
    :effect (and (not (at ?obj ?loc)) (in ?obj ?truck))
  )

  (:action load-airplane
    :parameters (?obj - obj ?airplane - airplane ?loc - airport)
    :precondition (and (loadairplane ?obj ?airplane ?loc) (at ?obj ?loc) (at ?airplane ?loc))
    :effect (and (not (at ?obj ?loc)) (in ?obj ?airplane))
  )

  (:action unload-truck
    :parameters (?obj - obj ?truck - truck ?loc - location)
    :precondition (and (unloadtruck ?obj ?truck ?loc) (at ?truck ?loc) (in ?obj ?truck))
    :effect (and (not (in ?obj ?truck)) (at ?obj ?loc))
  )

  (:action unload-airplane
    :parameters (?obj - obj ?airplane - airplane ?loc - airport)
    :precondition (and (unloadairplane ?obj ?airplane ?loc) (in ?obj ?airplane) (at ?airplane ?loc))
    :effect (and (not (in ?obj ?airplane)) (at ?obj ?loc))
  )

  (:action drive-truck
    :parameters (?truck - truck ?locfrom - location ?locto - location ?city - city)
    :precondition (and 
        (drivetruck ?truck ?locfrom ?locto ?city) 
        (at ?truck ?locfrom) 
        (in-city ?locfrom ?city) 
        (in-city ?locto ?city) 
        (different ?locfrom ?locto)        ;; replaced (not (= …))
    )
    :effect (and (not (at ?truck ?locfrom)) (at ?truck ?locto))
  )

  (:action fly-airplane
    :parameters (?airplane - airplane ?locfrom - airport ?locto - airport)
    :precondition (and 
        (flyairplane ?airplane ?locfrom ?locto) 
        (at ?airplane ?locfrom) 
        (different ?locfrom ?locto)        ;; replaced (not (= …))
    )
    :effect (and (not (at ?airplane ?locfrom)) (at ?airplane ?locto))
  )
)
