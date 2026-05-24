# Install script for directory: C:/Users/vaibh/OneDrive/Documents/Functional/baba-is-you/baba-is-auto/Libraries/doctest

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files/baba-is-auto")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/doctest" TYPE FILE FILES "C:/Users/vaibh/OneDrive/Documents/Functional/baba-is-you/baba-is-auto/Libraries/doctest/doctest/doctest.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/doctest" TYPE DIRECTORY FILES "C:/Users/vaibh/OneDrive/Documents/Functional/baba-is-you/baba-is-auto/Libraries/doctest/doctest/extensions" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/doctest" TYPE FILE FILES
    "C:/Users/vaibh/OneDrive/Documents/Functional/baba-is-you/baba-is-auto/build/temp.win-amd64-cpython-312/Release/Libraries/doctest/generated/doctestConfig.cmake"
    "C:/Users/vaibh/OneDrive/Documents/Functional/baba-is-you/baba-is-auto/build/temp.win-amd64-cpython-312/Release/Libraries/doctest/generated/doctestConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/doctest" TYPE FILE FILES
    "C:/Users/vaibh/OneDrive/Documents/Functional/baba-is-you/baba-is-auto/Libraries/doctest/scripts/cmake/doctest.cmake"
    "C:/Users/vaibh/OneDrive/Documents/Functional/baba-is-you/baba-is-auto/Libraries/doctest/scripts/cmake/doctestAddTests.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/doctest/doctestTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/doctest/doctestTargets.cmake"
         "C:/Users/vaibh/OneDrive/Documents/Functional/baba-is-you/baba-is-auto/build/temp.win-amd64-cpython-312/Release/Libraries/doctest/CMakeFiles/Export/fa5ba3fec28c3a20c06e45fe40ff2f72/doctestTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/doctest/doctestTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/doctest/doctestTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/doctest" TYPE FILE FILES "C:/Users/vaibh/OneDrive/Documents/Functional/baba-is-you/baba-is-auto/build/temp.win-amd64-cpython-312/Release/Libraries/doctest/CMakeFiles/Export/fa5ba3fec28c3a20c06e45fe40ff2f72/doctestTargets.cmake")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/vaibh/OneDrive/Documents/Functional/baba-is-you/baba-is-auto/build/temp.win-amd64-cpython-312/Release/Libraries/doctest/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
