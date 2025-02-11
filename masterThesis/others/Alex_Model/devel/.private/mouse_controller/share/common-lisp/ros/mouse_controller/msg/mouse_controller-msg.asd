
(cl:in-package :asdf)

(defsystem "mouse_controller-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "Floats" :depends-on ("_package_Floats"))
    (:file "_package_Floats" :depends-on ("_package"))
    (:file "Floats" :depends-on ("_package_Floats"))
    (:file "_package_Floats" :depends-on ("_package"))
    (:file "desired_cmd" :depends-on ("_package_desired_cmd"))
    (:file "_package_desired_cmd" :depends-on ("_package"))
    (:file "desired_cmd" :depends-on ("_package_desired_cmd"))
    (:file "_package_desired_cmd" :depends-on ("_package"))
    (:file "mouse_sensors" :depends-on ("_package_mouse_sensors"))
    (:file "_package_mouse_sensors" :depends-on ("_package"))
    (:file "mouse_sensors" :depends-on ("_package_mouse_sensors"))
    (:file "_package_mouse_sensors" :depends-on ("_package"))
  ))