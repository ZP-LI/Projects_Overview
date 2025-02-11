; Auto-generated. Do not edit!


(cl:in-package mouse_controller-msg)


;//! \htmlinclude desired_cmd.msg.html

(cl:defclass <desired_cmd> (roslisp-msg-protocol:ros-message)
  ((vel
    :reader vel
    :initarg :vel
    :type cl:float
    :initform 0.0)
   (turn_rate
    :reader turn_rate
    :initarg :turn_rate
    :type cl:float
    :initform 0.0)
   (buttons
    :reader buttons
    :initarg :buttons
    :type (cl:vector cl:integer)
   :initform (cl:make-array 0 :element-type 'cl:integer :initial-element 0)))
)

(cl:defclass desired_cmd (<desired_cmd>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <desired_cmd>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'desired_cmd)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name mouse_controller-msg:<desired_cmd> is deprecated: use mouse_controller-msg:desired_cmd instead.")))

(cl:ensure-generic-function 'vel-val :lambda-list '(m))
(cl:defmethod vel-val ((m <desired_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mouse_controller-msg:vel-val is deprecated.  Use mouse_controller-msg:vel instead.")
  (vel m))

(cl:ensure-generic-function 'turn_rate-val :lambda-list '(m))
(cl:defmethod turn_rate-val ((m <desired_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mouse_controller-msg:turn_rate-val is deprecated.  Use mouse_controller-msg:turn_rate instead.")
  (turn_rate m))

(cl:ensure-generic-function 'buttons-val :lambda-list '(m))
(cl:defmethod buttons-val ((m <desired_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mouse_controller-msg:buttons-val is deprecated.  Use mouse_controller-msg:buttons instead.")
  (buttons m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <desired_cmd>) ostream)
  "Serializes a message object of type '<desired_cmd>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'vel))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'turn_rate))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'buttons))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    ))
   (cl:slot-value msg 'buttons))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <desired_cmd>) istream)
  "Deserializes a message object of type '<desired_cmd>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'vel) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'turn_rate) (roslisp-utils:decode-single-float-bits bits)))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'buttons) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'buttons)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296)))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<desired_cmd>)))
  "Returns string type for a message object of type '<desired_cmd>"
  "mouse_controller/desired_cmd")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'desired_cmd)))
  "Returns string type for a message object of type 'desired_cmd"
  "mouse_controller/desired_cmd")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<desired_cmd>)))
  "Returns md5sum for a message object of type '<desired_cmd>"
  "6ca41f915100349b7a4af3c164f6abe2")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'desired_cmd)))
  "Returns md5sum for a message object of type 'desired_cmd"
  "6ca41f915100349b7a4af3c164f6abe2")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<desired_cmd>)))
  "Returns full string definition for message of type '<desired_cmd>"
  (cl:format cl:nil "float32 vel~%float32 turn_rate~%int32[] buttons~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'desired_cmd)))
  "Returns full string definition for message of type 'desired_cmd"
  (cl:format cl:nil "float32 vel~%float32 turn_rate~%int32[] buttons~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <desired_cmd>))
  (cl:+ 0
     4
     4
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'buttons) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <desired_cmd>))
  "Converts a ROS message object to a list"
  (cl:list 'desired_cmd
    (cl:cons ':vel (vel msg))
    (cl:cons ':turn_rate (turn_rate msg))
    (cl:cons ':buttons (buttons msg))
))
