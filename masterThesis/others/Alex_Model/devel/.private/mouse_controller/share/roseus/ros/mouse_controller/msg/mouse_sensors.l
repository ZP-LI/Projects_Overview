;; Auto-generated. Do not edit!


(when (boundp 'mouse_controller::mouse_sensors)
  (if (not (find-package "MOUSE_CONTROLLER"))
    (make-package "MOUSE_CONTROLLER"))
  (shadow 'mouse_sensors (find-package "MOUSE_CONTROLLER")))
(unless (find-package "MOUSE_CONTROLLER::MOUSE_SENSORS")
  (make-package "MOUSE_CONTROLLER::MOUSE_SENSORS"))

(in-package "ROS")
;;//! \htmlinclude mouse_sensors.msg.html


(defclass mouse_controller::mouse_sensors
  :super ros::object
  :slots (_servo_pos_leg _servo_pos_aux _contact_sensors _imu_sensor ))

(defmethod mouse_controller::mouse_sensors
  (:init
   (&key
    ((:servo_pos_leg __servo_pos_leg) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:servo_pos_aux __servo_pos_aux) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:contact_sensors __contact_sensors) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:imu_sensor __imu_sensor) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _servo_pos_leg __servo_pos_leg)
   (setq _servo_pos_aux __servo_pos_aux)
   (setq _contact_sensors __contact_sensors)
   (setq _imu_sensor __imu_sensor)
   self)
  (:servo_pos_leg
   (&optional __servo_pos_leg)
   (if __servo_pos_leg (setq _servo_pos_leg __servo_pos_leg)) _servo_pos_leg)
  (:servo_pos_aux
   (&optional __servo_pos_aux)
   (if __servo_pos_aux (setq _servo_pos_aux __servo_pos_aux)) _servo_pos_aux)
  (:contact_sensors
   (&optional __contact_sensors)
   (if __contact_sensors (setq _contact_sensors __contact_sensors)) _contact_sensors)
  (:imu_sensor
   (&optional __imu_sensor)
   (if __imu_sensor (setq _imu_sensor __imu_sensor)) _imu_sensor)
  (:serialization-length
   ()
   (+
    ;; float32[] _servo_pos_leg
    (* 4    (length _servo_pos_leg)) 4
    ;; float32[] _servo_pos_aux
    (* 4    (length _servo_pos_aux)) 4
    ;; float32[] _contact_sensors
    (* 4    (length _contact_sensors)) 4
    ;; float32[] _imu_sensor
    (* 4    (length _imu_sensor)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _servo_pos_leg
     (write-long (length _servo_pos_leg) s)
     (dotimes (i (length _servo_pos_leg))
       (sys::poke (elt _servo_pos_leg i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[] _servo_pos_aux
     (write-long (length _servo_pos_aux) s)
     (dotimes (i (length _servo_pos_aux))
       (sys::poke (elt _servo_pos_aux i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[] _contact_sensors
     (write-long (length _contact_sensors) s)
     (dotimes (i (length _contact_sensors))
       (sys::poke (elt _contact_sensors i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[] _imu_sensor
     (write-long (length _imu_sensor) s)
     (dotimes (i (length _imu_sensor))
       (sys::poke (elt _imu_sensor i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _servo_pos_leg
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _servo_pos_leg (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _servo_pos_leg i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[] _servo_pos_aux
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _servo_pos_aux (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _servo_pos_aux i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[] _contact_sensors
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _contact_sensors (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _contact_sensors i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[] _imu_sensor
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _imu_sensor (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _imu_sensor i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(setf (get mouse_controller::mouse_sensors :md5sum-) "7a3639d3aca39f506211e62a51b3df2b")
(setf (get mouse_controller::mouse_sensors :datatype-) "mouse_controller/mouse_sensors")
(setf (get mouse_controller::mouse_sensors :definition-)
      "float32[] servo_pos_leg
float32[] servo_pos_aux
float32[] contact_sensors
float32[] imu_sensor
")



(provide :mouse_controller/mouse_sensors "7a3639d3aca39f506211e62a51b3df2b")


