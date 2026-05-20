load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "drm_headers",
    visibility = ["//visibility:public"],
    deps = ["@libdrm//:drm_headers"],
)
