#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "libevent::core" for configuration "Release"
set_property(TARGET libevent::core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libevent::core PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libevent_core-2.1.7.dylib"
  IMPORTED_SONAME_RELEASE "/Users/adityasingh/Desktop/ml_project/human-activity-recognition/env/lib/libevent_core-2.1.7.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS libevent::core )
list(APPEND _IMPORT_CHECK_FILES_FOR_libevent::core "${_IMPORT_PREFIX}/lib/libevent_core-2.1.7.dylib" )

# Import target "libevent::extra" for configuration "Release"
set_property(TARGET libevent::extra APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libevent::extra PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libevent_extra-2.1.7.dylib"
  IMPORTED_SONAME_RELEASE "/Users/adityasingh/Desktop/ml_project/human-activity-recognition/env/lib/libevent_extra-2.1.7.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS libevent::extra )
list(APPEND _IMPORT_CHECK_FILES_FOR_libevent::extra "${_IMPORT_PREFIX}/lib/libevent_extra-2.1.7.dylib" )

# Import target "libevent::openssl" for configuration "Release"
set_property(TARGET libevent::openssl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libevent::openssl PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libevent_openssl-2.1.7.dylib"
  IMPORTED_SONAME_RELEASE "/Users/adityasingh/Desktop/ml_project/human-activity-recognition/env/lib/libevent_openssl-2.1.7.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS libevent::openssl )
list(APPEND _IMPORT_CHECK_FILES_FOR_libevent::openssl "${_IMPORT_PREFIX}/lib/libevent_openssl-2.1.7.dylib" )

# Import target "libevent::pthreads" for configuration "Release"
set_property(TARGET libevent::pthreads APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libevent::pthreads PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libevent_pthreads-2.1.7.dylib"
  IMPORTED_SONAME_RELEASE "/Users/adityasingh/Desktop/ml_project/human-activity-recognition/env/lib/libevent_pthreads-2.1.7.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS libevent::pthreads )
list(APPEND _IMPORT_CHECK_FILES_FOR_libevent::pthreads "${_IMPORT_PREFIX}/lib/libevent_pthreads-2.1.7.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
