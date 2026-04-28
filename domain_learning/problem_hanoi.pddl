(define (problem stack-problem)
  (:domain xyz)

  (:objects
    objpeg1 objpeg2 objpeg3
    objd1 objd2 objd3
    objrobot
  )

  (:init

    (clear objd1 objd1)
    (clear objd3 objd3)
    (clear objd2 objd2)


    (smaller objd1 objpeg2)
    (smaller objd1 objpeg1)
    (smaller objd1 objd3)
    (smaller objd1 objd2)
    (smaller objd1 objpeg3)

    (smaller objd3 objpeg2)
    (smaller objd3 objpeg1)
    (smaller objd3 objpeg3)

    (smaller objd2 objpeg3)
    (smaller objd2 objd3)
    (smaller objd2 objpeg2)
    (smaller objd2 objpeg1)


    (on objd1 objpeg1)
    (on objd3 objpeg2)
    (on objd2 objpeg3)

    
    (handempty objrobot objrobot)
  )

  (:goal (and
      (on objd1 objd2)
  ))
)
