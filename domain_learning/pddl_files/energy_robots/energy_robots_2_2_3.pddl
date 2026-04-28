(define (problem ER-2-2-3)
	(:domain energy_robots)
	(:objects
	c1 - ecell
	c2 - ecell
	c3 - ecell
	r1 - robot
	r2 - robot
	l1 - location
	l2 - location
	)
	(:init
	(at c1 l2)
	(at c2 l2)
	(at c3 l2)
	(at r1 l2)
	(at r2 l1)
    (handfree r1)
    (handfree r2)
)
	(:goal (and
	(at r2 l2)
	(used c1 r2)
	))


)
