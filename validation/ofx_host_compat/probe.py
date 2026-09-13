"""Exercise Juicer OFX lifecycle with optional host libraries preloaded."""

import argparse
import ctypes as c
import faulthandler
import hashlib
import json
import os
from pathlib import Path


faulthandler.enable()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("module", type=Path)
    parser.add_argument(
        "--preload",
        action="append",
        default=[],
        type=Path,
        help="Load a host library globally before loading the OFX module",
    )
    return parser.parse_args()


args = parse_args()
module_path = args.module.resolve()
preload_paths = [path.resolve() for path in args.preload]
pointer_type = c.c_void_p
string_type = c.c_char_p
int_type = c.c_int
double_type = c.c_double
refs = []
props = {
    1: {
        b"OfxPropType": [b"OfxTypeImageEffectHost"],
        b"OfxPropName": [b"FilmJuicerLinuxHostCompatibilityProbe"],
        b"OfxPropLabel": [b"Linux OFX host compatibility probe"],
        b"OfxPropAPIVersion": [1, 4],
        b"OfxPropVersion": [1, 0, 0],
        b"OfxPropVersionLabel": [b"1.0"],
        b"OfxImageEffectPropSupportedComponents": [b"OfxImageComponentRGBA"],
        b"OfxImageEffectPropSupportedContexts": [b"OfxImageEffectContextFilter"],
        b"OfxImageEffectPropSupportedPixelDepths": [b"OfxBitDepthFloat"],
        b"OfxImageEffectPropCudaRenderSupported": [b"true"],
        b"OfxImageEffectPropCudaStreamSupported": [b"true"],
        b"OfxImageEffectPropOpenCLRenderSupported": [b"false"],
        b"OfxImageEffectPropMetalRenderSupported": [b"false"],
        b"OfxImageEffectHostPropNativeOrigin": [b"OfxHostNativeOriginBottomLeft"],
    },
    2: {b"OfxPropType": [b"OfxTypeImageEffect"]},
    3: {},
}


def callback(result, arguments, function):
    value = c.CFUNCTYPE(result, *arguments)(function)
    refs.append(value)
    return c.cast(value, pointer_type).value


def set_value(handle, key, index, value):
    values = props.setdefault(handle, {}).setdefault(key, [])
    while len(values) <= index:
        values.append(None)
    values[index] = value
    return 0


def get_value(handle, key, index, output, value_type):
    values = props.get(handle, {}).get(key)
    if handle == 1 and values is None and value_type in (int_type, double_type):
        values = [0, 0]
    if values is None or index >= len(values):
        return 3
    output[0] = values[index]
    return 0


def set_multiple(handle, key, count, values):
    for index in range(count):
        result = set_value(handle, key, index, values[index])
        if result:
            return result
    return 0


slots = []
for is_get in (False, True):
    for is_many in (False, True):
        for value_type in (pointer_type, string_type, double_type, int_type):
            if is_get:

                def function(handle, key, count, output, value_type=value_type, many=is_many):
                    for index in range(count if many else 1):
                        offset = index * c.sizeof(value_type)
                        pointer = c.cast(c.byref(output.contents, offset), c.POINTER(value_type))
                        result = get_value(handle, key, index if many else count, pointer, value_type)
                        if result:
                            return result
                    return 0

            elif is_many:
                function = set_multiple
            else:
                function = set_value
            argument_type = c.POINTER(value_type) if is_get or is_many else value_type
            slots.append(callback(int_type, [pointer_type, string_type, int_type, argument_type], function))
slots.append(
    callback(
        int_type,
        [pointer_type, string_type],
        lambda handle, key: (props[handle].pop(key, None), 0)[1],
    )
)


def dimension(handle, key, output):
    output[0] = len(props.get(handle, {}).get(key, []))
    return 0


slots.append(callback(int_type, [pointer_type, string_type, c.POINTER(int_type)], dimension))
property_suite = (pointer_type * len(slots))(*slots)


def get_handle(handle, output):
    output[0] = handle
    return 0


def get_params(handle, output):
    output[0] = 3
    return 0


image_suite = (pointer_type * 13)(
    callback(int_type, [pointer_type, c.POINTER(pointer_type)], get_handle),
    callback(int_type, [pointer_type, c.POINTER(pointer_type)], get_params),
)
parameter_suite = (pointer_type * 32)()
parameter_suite[2] = callback(int_type, [pointer_type, c.POINTER(pointer_type)], get_handle)
dummy_suite = (pointer_type * 32)()
suite_requests = []


def fetch(_handle, name, version):
    suite_requests.append([name.decode(), version])
    if name == b"OfxPropertySuite":
        return c.addressof(property_suite)
    if name == b"OfxImageEffectSuite":
        return c.addressof(image_suite)
    if name == b"OfxParameterSuite":
        return c.addressof(parameter_suite)
    if name in (b"OfxMemorySuite", b"OfxMultiThreadSuite", b"OfxMessageSuite"):
        return c.addressof(dummy_suite)
    return None


class Host(c.Structure):
    _fields_ = [("host", pointer_type), ("fetchSuite", pointer_type)]


class Plugin(c.Structure):
    _fields_ = [
        ("api", string_type),
        ("apiVersion", int_type),
        ("identifier", string_type),
        ("major", c.c_uint),
        ("minor", c.c_uint),
        ("setHost", pointer_type),
        ("mainEntry", pointer_type),
    ]


host = Host(1, callback(pointer_type, [pointer_type, string_type, int_type], fetch))
preloaded = [c.CDLL(str(path), mode=os.RTLD_NOW | os.RTLD_GLOBAL) for path in preload_paths]
module = c.CDLL(str(module_path), mode=os.RTLD_NOW | os.RTLD_LOCAL)
module.OfxGetNumberOfPlugins.restype = int_type
module.OfxGetPlugin.argtypes = [int_type]
module.OfxGetPlugin.restype = c.POINTER(Plugin)
assert module.OfxGetNumberOfPlugins() == 1
plugin = module.OfxGetPlugin(0).contents
c.CFUNCTYPE(None, c.POINTER(Host))(plugin.setHost)(c.byref(host))
identifier = plugin.identifier.decode()
entry = c.CFUNCTYPE(int_type, string_type, pointer_type, pointer_type, pointer_type)(plugin.mainEntry)
results = {}
for action, handle in (
    (b"OfxActionLoad", None),
    (b"OfxActionDescribe", 2),
    (b"OfxActionUnload", None),
):
    print("Running " + action.decode(), flush=True)
    results[action.decode()] = entry(action, handle, None, None)
    if results[action.decode()] != 0:
        break

print(
    json.dumps(
        {
            "module": str(module_path),
            "sha256": hashlib.file_digest(module_path.open("rb"), "sha256").hexdigest(),
            "preloaded": [str(path) for path in preload_paths],
            "identifier": identifier,
            "actions": results,
            "suite_requests": suite_requests,
            "scope": "Synthetic host lifecycle with optional host libraries; no rendering",
        },
        indent=2,
    )
)
assert results == {
    "OfxActionLoad": 0,
    "OfxActionDescribe": 0,
    "OfxActionUnload": 0,
}, results
assert props[2][b"OfxImageEffectPropCudaRenderSupported"] == [b"true"]
