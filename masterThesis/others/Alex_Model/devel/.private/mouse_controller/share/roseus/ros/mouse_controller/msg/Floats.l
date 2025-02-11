;; Auto-generated. Do not edit!


(when (boundp 'mouse_controller::Floats)
  (if (not (find-package "MOUSE_CONTROLLER"))
    (make-package "MOUSE_CONTROLLER"))
  (shadow 'Floats (find-package "MOUSE_CONTROLLER")))
(unless (find-package "MOUSE_CONTROLLER::FLOATS")
  (make-package "MOUSE_CONTROLLER::FLOATS"))

(in-package "ROS")
;;//! \htmlinclude Floats.msg.html


(defclass mouse_controller::Floats
  :super ros::object
  :slots (_data ))

(defmethod mouse_controller::Floats
  (:init
   (&key
    ((:data __data) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _data __data)
   self)
  (:data
   (&optional __data)
   (if __data (setq _data __data)) _data)
  (:serialization-length
   ()
   (+
    ;; float32[] _data
    (* 4    (length _data)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _data
     (write-long (length _data) s)
     (dotimes (i (length _data))
       (sys::poke (elt _data i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _data
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _data (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _data i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(setf (get mouse_controller::Floats :md5sum-) "420cd38b6b071cd49f2970c3e2cee511")
(setf (get mouse_controller::Floats :datatype-) "mouse_controller/Floats")
(setf (get mouse_controller::Floats :definition-)
      "float32[] data
")



(provide :mouse_controller/Floats "420cd38b6b071cd49f2970c3e2cee511")


