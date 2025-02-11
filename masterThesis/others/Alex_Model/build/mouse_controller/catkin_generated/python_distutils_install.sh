#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/zhiping/MA/Alex_Model/src/mouse_controller"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/zhiping/MA/Alex_Model/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/zhiping/MA/Alex_Model/install/lib/python3/dist-packages:/home/zhiping/MA/Alex_Model/build/mouse_controller/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/zhiping/MA/Alex_Model/build/mouse_controller" \
    "/usr/bin/python3" \
    "/home/zhiping/MA/Alex_Model/src/mouse_controller/setup.py" \
     \
    build --build-base "/home/zhiping/MA/Alex_Model/build/mouse_controller" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/zhiping/MA/Alex_Model/install" --install-scripts="/home/zhiping/MA/Alex_Model/install/bin"
