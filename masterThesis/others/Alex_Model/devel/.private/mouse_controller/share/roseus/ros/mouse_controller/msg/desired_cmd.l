;; Auto-generated. Do not edit!


(when (boundp 'mouse_controller::desired_cmd)
  (if (not (find-package "MOUSE_CONTROLLER"))
    (make-package "MOUSE_CONTROLLER"))
  (shadow 'desired_cmd (find-package "MOUSE_CONTROLLER")))
(unless (find-package "MOUSE_CONTROLLER::DESIRED_CMD")
  (make-package "MOUSE_CONTROLLER::DESIRED_CMD"))

(in-package "ROS")
;;//! \htmlinclude desired_cmd.msg.html


(defclass mouse_controller::desired_cmd
  :super ros::object
  :slots (_vel _turn_rate _buttons ))

(defmethod mouse_controller::desired_cmd
  (:init
   (&key
    ((:vel __vel) 0.0)
    ((:turn_rate __turn_rate) 0.0)
    ((:buttons __buttons) (make-array 0 :initial-element 0 :element-type :integer))
    )
   (send-super :init)
   (setq _vel (float __vel))
   (setq _turn_rate (float __turn_rate))
   (setq _buttons __buttons)
   self)
  (:vel
   (&optional __vel)
   (if __vel (setq _vel __vel)) _vel)
  (:turn_rate
   (&optional __turn_rate)
   (if __turn_rate (setq _turn_rate __turn_rate)) _turn_rate)
  (:buttons
   (&optional __buttons)
   (if __buttons (setq _buttons __buttons)) _buttons)
  (:serialization-length
   ()
   (+
    ;; float32 _vel
    4
    ;; float32 _turn_rate
    4
    ;; int32[] _buttons
    (* 4    (length _buttons)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _vel
       (sys::poke _vel (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _turn_rate
       (sys::poke _turn_rate (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; int32[] _buttons
     (write-long (length _buttons) s)
     (dotimes (i (length _buttons))
       (write-long (elt _buttons i) s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _vel
     (setq _vel (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _turn_rate
     (setq _turn_rate (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; int32[] _buttons
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _buttons (instantiate integer-vector n))
     (dotimes (i n)
     (setf (elt _buttons i) (sys::peek buf ptr- :integer)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(setf (get mouse_controller::desired_cmd :md5sum-) "6ca41f915100349b7a4af3c164f6abe2")
(setf (get mouse_controller::desired_cmd :datatype-) "mouse_controller/desired_cmd")
(setf (get mouse_controller::desired_cmd :definition-)
      "float32 vel
float32 turn_rate
int32[] buttons
")



(provide :mouse_controller/desired_cmd "6ca41f915100349b7a4af3c164f6abe2")


