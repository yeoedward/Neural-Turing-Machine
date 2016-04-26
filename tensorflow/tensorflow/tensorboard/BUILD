# Description:
# TensorBoard, a dashboard for investigating TensorFlow

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "frontend",
    srcs = [
        "dist/index.html",
        "dist/tf-tensorboard.html",
        "TAG",
        "//tensorflow/tensorboard/bower:bower",
        "//tensorflow/tensorboard/lib:all_files",
    ],
)

py_binary(
    name = "tensorboard",
    srcs = ["tensorboard.py"],
    data = [":frontend"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:platform",
        "//tensorflow/tensorboard/backend:server",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "**/node_modules/**",
            "**/typings/**",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
