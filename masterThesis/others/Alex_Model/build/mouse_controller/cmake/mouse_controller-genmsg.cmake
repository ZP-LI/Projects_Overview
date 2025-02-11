# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "mouse_controller: 3 messages, 0 services")

set(MSG_I_FLAGS "-Imouse_controller:/home/zhiping/MA/Alex_Model/src/mouse_controller/msg;-Imouse_controller:/home/zhiping/MA/Alex_Model/src/mouse_controller/msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(mouse_controller_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/Floats.msg" NAME_WE)
add_custom_target(_mouse_controller_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "mouse_controller" "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/Floats.msg" ""
)

get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/mouse_sensors.msg" NAME_WE)
add_custom_target(_mouse_controller_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "mouse_controller" "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/mouse_sensors.msg" ""
)

get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/desired_cmd.msg" NAME_WE)
add_custom_target(_mouse_controller_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "mouse_controller" "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/desired_cmd.msg" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(mouse_controller
  "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/Floats.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/mouse_controller
)
_generate_msg_cpp(mouse_controller
  "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/mouse_sensors.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/mouse_controller
)
_generate_msg_cpp(mouse_controller
  "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/desired_cmd.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/mouse_controller
)

### Generating Services

### Generating Module File
_generate_module_cpp(mouse_controller
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/mouse_controller
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(mouse_controller_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(mouse_controller_generate_messages mouse_controller_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/Floats.msg" NAME_WE)
add_dependencies(mouse_controller_generate_messages_cpp _mouse_controller_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/mouse_sensors.msg" NAME_WE)
add_dependencies(mouse_controller_generate_messages_cpp _mouse_controller_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/desired_cmd.msg" NAME_WE)
add_dependencies(mouse_controller_generate_messages_cpp _mouse_controller_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(mouse_controller_gencpp)
add_dependencies(mouse_controller_gencpp mouse_controller_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS mouse_controller_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(mouse_controller
  "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/Floats.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/mouse_controller
)
_generate_msg_eus(mouse_controller
  "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/mouse_sensors.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/mouse_controller
)
_generate_msg_eus(mouse_controller
  "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/desired_cmd.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/mouse_controller
)

### Generating Services

### Generating Module File
_generate_module_eus(mouse_controller
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/mouse_controller
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(mouse_controller_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(mouse_controller_generate_messages mouse_controller_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/Floats.msg" NAME_WE)
add_dependencies(mouse_controller_generate_messages_eus _mouse_controller_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/mouse_sensors.msg" NAME_WE)
add_dependencies(mouse_controller_generate_messages_eus _mouse_controller_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/desired_cmd.msg" NAME_WE)
add_dependencies(mouse_controller_generate_messages_eus _mouse_controller_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(mouse_controller_geneus)
add_dependencies(mouse_controller_geneus mouse_controller_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS mouse_controller_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(mouse_controller
  "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/Floats.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/mouse_controller
)
_generate_msg_lisp(mouse_controller
  "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/mouse_sensors.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/mouse_controller
)
_generate_msg_lisp(mouse_controller
  "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/desired_cmd.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/mouse_controller
)

### Generating Services

### Generating Module File
_generate_module_lisp(mouse_controller
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/mouse_controller
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(mouse_controller_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(mouse_controller_generate_messages mouse_controller_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/Floats.msg" NAME_WE)
add_dependencies(mouse_controller_generate_messages_lisp _mouse_controller_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/mouse_sensors.msg" NAME_WE)
add_dependencies(mouse_controller_generate_messages_lisp _mouse_controller_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/desired_cmd.msg" NAME_WE)
add_dependencies(mouse_controller_generate_messages_lisp _mouse_controller_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(mouse_controller_genlisp)
add_dependencies(mouse_controller_genlisp mouse_controller_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS mouse_controller_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(mouse_controller
  "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/Floats.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/mouse_controller
)
_generate_msg_nodejs(mouse_controller
  "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/mouse_sensors.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/mouse_controller
)
_generate_msg_nodejs(mouse_controller
  "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/desired_cmd.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/mouse_controller
)

### Generating Services

### Generating Module File
_generate_module_nodejs(mouse_controller
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/mouse_controller
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(mouse_controller_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(mouse_controller_generate_messages mouse_controller_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/Floats.msg" NAME_WE)
add_dependencies(mouse_controller_generate_messages_nodejs _mouse_controller_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/mouse_sensors.msg" NAME_WE)
add_dependencies(mouse_controller_generate_messages_nodejs _mouse_controller_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/desired_cmd.msg" NAME_WE)
add_dependencies(mouse_controller_generate_messages_nodejs _mouse_controller_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(mouse_controller_gennodejs)
add_dependencies(mouse_controller_gennodejs mouse_controller_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS mouse_controller_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(mouse_controller
  "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/Floats.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/mouse_controller
)
_generate_msg_py(mouse_controller
  "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/mouse_sensors.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/mouse_controller
)
_generate_msg_py(mouse_controller
  "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/desired_cmd.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/mouse_controller
)

### Generating Services

### Generating Module File
_generate_module_py(mouse_controller
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/mouse_controller
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(mouse_controller_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(mouse_controller_generate_messages mouse_controller_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/Floats.msg" NAME_WE)
add_dependencies(mouse_controller_generate_messages_py _mouse_controller_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/mouse_sensors.msg" NAME_WE)
add_dependencies(mouse_controller_generate_messages_py _mouse_controller_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zhiping/MA/Alex_Model/src/mouse_controller/msg/desired_cmd.msg" NAME_WE)
add_dependencies(mouse_controller_generate_messages_py _mouse_controller_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(mouse_controller_genpy)
add_dependencies(mouse_controller_genpy mouse_controller_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS mouse_controller_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/mouse_controller)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/mouse_controller
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET mouse_controller_generate_messages_cpp)
  add_dependencies(mouse_controller_generate_messages_cpp mouse_controller_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/mouse_controller)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/mouse_controller
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET mouse_controller_generate_messages_eus)
  add_dependencies(mouse_controller_generate_messages_eus mouse_controller_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/mouse_controller)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/mouse_controller
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET mouse_controller_generate_messages_lisp)
  add_dependencies(mouse_controller_generate_messages_lisp mouse_controller_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/mouse_controller)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/mouse_controller
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET mouse_controller_generate_messages_nodejs)
  add_dependencies(mouse_controller_generate_messages_nodejs mouse_controller_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/mouse_controller)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/mouse_controller\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/mouse_controller
    DESTINATION ${genpy_INSTALL_DIR}
    # skip all init files
    PATTERN "__init__.py" EXCLUDE
    PATTERN "__init__.pyc" EXCLUDE
  )
  # install init files which are not in the root folder of the generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/mouse_controller
    DESTINATION ${genpy_INSTALL_DIR}
    FILES_MATCHING
    REGEX "${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/mouse_controller/.+/__init__.pyc?$"
  )
endif()
if(TARGET mouse_controller_generate_messages_py)
  add_dependencies(mouse_controller_generate_messages_py mouse_controller_generate_messages_py)
endif()
