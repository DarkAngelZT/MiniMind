#!/usr/bin/env python
import hashlib
import os
import subprocess

Import("env")

# MiniMind 在自己的构建入口中封装 MNN、CMake 和平台相关细节。
platform_name = env["platform"]
target_name = env["target"]
cmake_build_type = "Debug" if target_name == "template_debug" else "Release"
parallel_jobs = max(1, int(GetOption("num_jobs") or 1))

minimind_source_dir = Dir(".").abspath
minimind_build_dir = os.path.join(
    minimind_source_dir, "build", "scons", platform_name, target_name
)
configure_stamp = os.path.join(minimind_build_dir, ".scons_configure.stamp")
build_stamp = os.path.join(minimind_build_dir, ".scons_build.stamp")


def collect_inputs(configuration_only=False):
    """收集会影响配置或编译的文件，忽略构建输出和单元测试目录。"""
    result = []
    source_extensions = {
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
        ".s", ".asm", ".in",
    }
    ignored_directories = {".git", "build", "tests"}

    for root, directories, files in os.walk(minimind_source_dir):
        directories[:] = [
            name for name in directories if name not in ignored_directories
        ]
        for name in files:
            extension = os.path.splitext(name)[1].lower()
            is_configuration = name == "CMakeLists.txt" or extension == ".cmake"
            if is_configuration or (not configuration_only and extension in source_extensions):
                result.append(os.path.join(root, name))
    return sorted(result)


def run_checked(command):
    """执行外部构建命令，并将失败状态传递给 SCons。"""
    print(" ".join(command))
    subprocess.run(command, check=True)


def hash_files(paths, extra_values):
    """计算配置摘要，使内容变化能够正确传递给后续构建节点。"""
    digest = hashlib.sha256()
    for value in extra_values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for path in sorted(paths):
        digest.update(path.encode("utf-8"))
        with open(path, "rb") as input_file:
            for block in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def configure_minimind(target, source, env):
    """配置 MiniMind，并根据父工程平台选择一致的编译工具链。"""
    os.makedirs(minimind_build_dir, exist_ok=True)
    command = [
        "cmake",
        "-S", minimind_source_dir,
        "-B", minimind_build_dir,
        "-DCMAKE_BUILD_TYPE=" + cmake_build_type,
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DMINIMIND_BUILD_TESTS=OFF",
    ]

    if platform_name == "windows":
        command.extend([
            "-G", "MinGW Makefiles",
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
            "-DCMAKE_MAKE_PROGRAM=mingw32-make",
        ])
    elif platform_name == "linux":
        command.extend([
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
        ])

    run_checked(command)
    configuration_hash = hash_files([str(node) for node in source], command)
    with open(str(target[0]), "w", encoding="utf-8") as stamp:
        stamp.write(configuration_hash + "\n")
    return 0


def build_minimind(target, source, env):
    """增量构建 MiniMind 目标，其依赖的 MNN 会由 CMake 一并生成。"""
    run_checked([
        "cmake",
        "--build", minimind_build_dir,
        "--target", "MiniMind",
        "--parallel", str(parallel_jobs),
    ])

    minimind_library = os.path.join(minimind_build_dir, "libMiniMind.a")
    mnn_library = os.path.join(minimind_build_dir, "mnn", "libMNN.a")
    libraries = (minimind_library, mnn_library)
    missing = [path for path in libraries if not os.path.isfile(path)]
    if missing:
        raise RuntimeError("MiniMind 构建完成后缺少静态库：" + "、".join(missing))

    library_state = [
        "{}:{}:{}".format(path, os.path.getsize(path), os.stat(path).st_mtime_ns)
        for path in libraries
    ]
    with open(str(target[0]), "w", encoding="utf-8") as stamp:
        stamp.write("\n".join(library_state) + "\n")
    return 0


configure_node = env.Command(
    configure_stamp,
    collect_inputs(configuration_only=True),
    Action(configure_minimind, "正在配置 MiniMind（${TARGET}）..."),
)
build_node = env.Command(
    build_stamp,
    collect_inputs(configuration_only=False) + configure_node,
    Action(build_minimind, "正在构建 MiniMind 和 MNN（${TARGET}）..."),
)

# 调用方只需导入本入口，所需头文件、库目录和链接库会自动附加。
env.Append(CPPPATH=[
    os.path.join(minimind_source_dir, "src"),
    os.path.join(minimind_source_dir, "src", "3rd_party", "MNN", "include"),
    os.path.join(minimind_source_dir, "src", "3rd_party", "MNN", "schema", "current"),
])
env.Append(LIBPATH=[
    minimind_build_dir,
    os.path.join(minimind_build_dir, "mnn"),
])
env.Append(LIBS=["MiniMind", "MNN", "pthread"])

Return("build_node")
