cmake_minimum_required(VERSION 3.18)
project(AlphaZeroDotBoxes LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── LibTorch ─────────────────────────────────────────────────────────
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/include/libtorch")
find_package(Torch REQUIRED)

# ─── Include directories ─────────────────────────────────────────────
include_directories(
    ${CMAKE_SOURCE_DIR}/include        # BitBoardEnv.h, FastBitSet.h
    ${CMAKE_SOURCE_DIR}/agents         # AlphaZeroBitAgent.h
    ${CMAKE_SOURCE_DIR}/models         # AlphaZeroBitNet.h
    ${CMAKE_SOURCE_DIR}/train          # AlphaZeroTrain.h, TrainConfig.h, etc.
)

# ─── Core source files ───────────────────────────────────────────────
set(CORE_SOURCES
    env/BitBoardEnv.cpp
    agents/AlphaZeroBitAgent.cpp
)

# ─── alphazero_train ──────────────────────────────────────────────────
add_executable(alphazero_train
    ${CORE_SOURCES}
    train/AlphaZeroTrain.cpp
    main.cpp
)
target_link_libraries(alphazero_train ${TORCH_LIBRARIES} pthread)
set_property(TARGET alphazero_train PROPERTY CXX_STANDARD 17)
target_compile_options(alphazero_train PRIVATE ${TORCH_CXX_FLAGS} -march=native -mtune=native)

# Link-time optimization for cross-file inlining (critical for header-only FastBitset/etc.)
include(CheckIPOSupported)
check_ipo_supported(RESULT lto_supported)
if(lto_supported)
    set_property(TARGET alphazero_train PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ─── alphazero_server ─────────────────────────────────────────────────
# Stdin/stdout agent server for Python interop.
# Python spawns this process and sends game states, receives actions.
add_executable(alphazero_server
    ${CORE_SOURCES}
    server.cpp
)
target_link_libraries(alphazero_server ${TORCH_LIBRARIES} pthread)
set_property(TARGET alphazero_server PROPERTY CXX_STANDARD 17)
target_compile_options(alphazero_server PRIVATE ${TORCH_CXX_FLAGS} -march=native -mtune=native)
if(lto_supported)
    set_property(TARGET alphazero_server PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
