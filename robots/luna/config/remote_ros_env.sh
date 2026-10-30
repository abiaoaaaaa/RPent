#!/usr/bin/env bash
# Purpose: load the minimal local ROS overlay and discover Luna at 6.6.7.100.
# Usage: source robots/luna/config/remote_ros_env.sh

unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE _CE_CONDA _CE_M
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

source /opt/ros/humble/setup.bash

LUNA_ROS_OVERLAY="${LUNA_ROS_OVERLAY:-/home/xlb/data/code/code_new/Luna/rpent_ros_overlay_clean/install/setup.bash}"
if [[ ! -f "${LUNA_ROS_OVERLAY}" ]]; then
  echo "Luna ROS overlay is missing: ${LUNA_ROS_OVERLAY}" >&2
  return 1
fi
source "${LUNA_ROS_OVERLAY}"

export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/cyclonedds_remote.xml"
export LUNA_APEX_WORKSPACE="${LUNA_APEX_WORKSPACE:-/home/xlb/data/code/code_new/Luna/Apex_Deploy}"

# The portable Apex install tree contains Python symlinks into the original
# developer workspace.  The colcon build tree in LUNA_APEX_WORKSPACE has the
# same generated move/server interfaces and remains valid after copying the
# workspace, so expose those packages to the ROS-enabled system Python.
if [[ -d "${LUNA_APEX_WORKSPACE}/build" ]]; then
  while IFS= read -r generated_python_dir; do
    export PYTHONPATH="${generated_python_dir}${PYTHONPATH:+:${PYTHONPATH}}"
  done < <(find "${LUNA_APEX_WORKSPACE}/build" -type d -name rosidl_generator_py -print)
fi

if [[ -d "${LUNA_APEX_WORKSPACE}/install" ]]; then
  while IFS= read -r package_lib_dir; do
    export LD_LIBRARY_PATH="${package_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  done < <(find "${LUNA_APEX_WORKSPACE}/install" -type d -name lib -print)
fi

export NO_PROXY="${NO_PROXY:+${NO_PROXY},}6.6.7.100"
export no_proxy="${no_proxy:+${no_proxy},}6.6.7.100"
unset LUNA_ROS_OVERLAY
