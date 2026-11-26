"""Observe scanner capacity and replay in symbol-matched LLDB targets."""

import hashlib
import json
import os
import struct
import subprocess
import time
from pathlib import Path

import lldb

WINDOWS = os.name == "nt"
POINTS = {
    "render": "CallScope::render",
    "spectral_launch": "scan_linear_density_rgb_kernel",
    "allocate": "allocate_owned_device",
    "upload": "enqueue_host_to_device_copy",
    "retain": "retain_scan_error_readback",
    "poll": "poll_scan_error_readbacks",
    "query": "cudaEventQuery",
    "destroy_event": "cudaEventDestroy",
}
REPLAY_POINTS = {
    **{name: POINTS[name] for name in ("render", "spectral_launch", "retain", "poll", "query", "destroy_event")},
    "prepare": "prepare_scan_error_stage",
    "current_grain": "juicer_cuda_apply_visual_grain",
}
TAIL_POINTS = {name: REPLAY_POINTS[name] for name in ("render", "prepare", "retain", "poll", "query", "destroy_event")}
PHOTO_MODES = ("print-photo", "current-photo")
SCAN_ROUTES = (
    ("NegativeDirectScan", "Direct"),
    ("NegativePrintScan", "Print"),
    ("PositiveDirectScan", "Direct"),
    ("PositivePrintScan", "Print"),
)
settings = {}
entries = {}
returns = {}
frames = []
active = {}
errors = []
tail_predecessor = None


def scan_medium(tag, expected):
    if type(tag) is not int or not 0 <= tag < len(SCAN_ROUTES):
        raise ValueError(f"unknown native scanner route {tag}")
    route, medium = SCAN_ROUTES[tag]
    if route != expected:
        raise ValueError(f"expected scanner route {expected}, observed {route}")
    return medium


def record(observation, **fields):
    if "event" in fields:
        fields["cuda_event"] = fields.pop("event")
    with Path(settings["trace"]).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": observation, **fields}, sort_keys=True) + "\n")


def read(frame, address, size=8):
    error = lldb.SBError()
    value = frame.GetThread().GetProcess().ReadUnsignedFromMemory(address, size, error)
    if error.Fail():
        raise RuntimeError(f"cannot read {size} bytes at {address:#x}: {error}")
    return value


def c_string(frame, address):
    error = lldb.SBError()
    value = frame.GetThread().GetProcess().ReadCStringFromMemory(address, 256, error)
    if error.Fail() or not value or len(value) >= 255:
        raise RuntimeError(f"cannot read label at {address:#x}: {error}")
    return value


def register(frame, name):
    value = frame.FindRegister(name)
    if not value.IsValid() or value.GetError().Fail():
        raise RuntimeError(f"register unavailable: {name}")
    return value.GetValueAsUnsigned()


def argument(frame, index):
    names = ("rcx", "rdx", "r8", "r9") if WINDOWS else ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
    if index < len(names):
        return register(frame, names[index])
    stack_index = index + 1 if WINDOWS else index - 5
    return read(frame, register(frame, "rsp") + 8 * stack_index)


def at_return(frame, handler):
    thread = frame.GetThread()
    address = read(frame, register(frame, "rsp"))
    breakpoint = thread.GetProcess().GetTarget().BreakpointCreateByAddress(address)
    if not breakpoint.IsValid() or breakpoint.GetNumLocations() != 1:
        raise RuntimeError(f"unavailable return breakpoint at {address:#x}")
    breakpoint.SetOneShot(True)
    breakpoint.SetThreadID(thread.GetThreadID())
    breakpoint.SetScriptCallbackFunction("observe_seed.on_return")
    returns[breakpoint.GetID()] = handler


def fail(error):
    errors.append(str(error))
    record("observation_error", detail=str(error))


def on_return(frame, location, _):
    try:
        breakpoint = location.GetBreakpoint()
        breakpoint.SetEnabled(False)
        handler = returns.pop(breakpoint.GetID(), None)
        if handler is not None:
            handler(frame)
    except Exception as error:
        fail(error)
    return False


def render(frame, thread):
    if thread in active:
        raise RuntimeError("unexpected reentrant capacity render")
    memory_result = argument(frame, 0) if WINDOWS else None
    shift = int(WINDOWS)
    context = argument(frame, 1 + shift)
    raster = argument(frame, 2 + shift)
    submission = argument(frame, 3 + shift)
    prepared = argument(frame, 4 + shift)
    draw = {
        "index": len(frames),
        "thread": thread,
        "device": read(frame, context, 4),
        "context": read(frame, context + 8),
        "stream": read(frame, raster + 0x70),
        "time_bits": read(frame, raster + 0x80),
        "instance": read(frame, submission),
        "token": read(frame, submission + 8),
        "submission": read(frame, submission + 16),
        "route": read(frame, prepared, 4),
        "allocations": [],
        "uploads": [],
        "launches": [],
    }
    if "expected_route" in settings:
        draw["scan_medium"] = scan_medium(draw["route"], settings["expected_route"])
    if settings.get("mode") == "tail":
        draw["native_handle"] = read(frame, argument(frame, shift) + settings["layout"]["call_scope_owner"])
        draw["bindings"] = [read(frame, submission + offset) for offset in (24, 32, 40, 48)]
        if settings.get("replay_tokens"):
            draw["tokens"] = [read(frame, raster + 0x90), draw["instance"], read(frame, raster + 0x98)]
            draw["current_grain"] = None
    if settings.get("mode") == "replay":
        # FjFrame and FjPreparedHostData layouts are fixed by the native ABI.
        film = read(frame, prepared + 6704)
        draw["tokens"] = [read(frame, raster + 0x90), draw["instance"], read(frame, raster + 0x98)]
        draw["film_grain"] = [read(frame, film + offset, 4) for offset in (680, 684, 692)] if film else None
        draw["current_grain"] = None
    if settings.get("mode") == "current-photo":
        draw["tokens"] = [read(frame, raster + 0x90), draw["instance"], read(frame, raster + 0x98)]
        draw["current_grain"] = None
    frames.append(draw)
    active[thread] = draw
    record("render_binding", **draw)

    def finished(return_frame):
        if WINDOWS:
            draw["outcome"] = [read(return_frame, memory_result + offset, 4) for offset in (0, 4, 8, 12)]
        else:
            words = [register(return_frame, name) for name in ("rax", "rdx")]
            draw["outcome"] = [part for word in words for part in (word & 0xFFFFFFFF, word >> 32)]
        del active[thread]
        record("render_return", index=draw["index"], outcome=draw["outcome"])

    at_return(frame, finished)


def owner(frame, address):
    return {
        "resources": address,
        "device": read(frame, address, 4),
        "context": read(frame, address + 8),
        "epoch": read(frame, address + 16),
    }


def retain(frame, draw):
    if "retained" in draw:
        raise RuntimeError("more than one retained readback for a capacity render")
    host_ref = argument(frame, 1)
    event_ref = argument(frame, 2)
    held = {
        **owner(frame, argument(frame, 0)),
        "host": read(frame, host_ref),
        "event": read(frame, event_ref),
        "epoch_at_production": read(frame, argument(frame, 3) + 16),
    }
    if settings.get("mode") in ("replay", "tail", *PHOTO_MODES):
        identity = argument(frame, 3)
        held["producer"] = [read(frame, identity + offset) for offset in settings["layout"]["producer"]]
    draw["retained"] = held

    def finished(return_frame):
        held["success"] = register(return_frame, "rax") & 0xFF == 1
        held["handoff"] = read(return_frame, host_ref) == 0 and read(return_frame, event_ref) == 0
        record("readback_retained", index=draw["index"], **held)

    at_return(frame, finished)


def readback_range(frame, address):
    vector = address + settings["layout"]["pending_vector"]
    return [read(frame, vector), read(frame, vector + 8)]


def poll(frame, draw):
    if "poll" in draw:
        raise RuntimeError("more than one scan poll for a capacity render")
    # The archived cubic poll takes the reporter stream. The spectral poll
    # only queries events. The bound render supplies its stream, and replay
    # and tail independently check it at scan preparation.
    takes_stream = settings.get("layout", {}).get("poll_takes_stream", True)
    stream = argument(frame, 1) if takes_stream else draw["stream"]
    polled = {**owner(frame, argument(frame, 0)), "stream": stream, "queries": [], "destroyed": []}
    draw["poll"] = polled
    result = argument(frame, 2 if takes_stream else 1)
    if settings.get("mode") in ("replay", "tail", *PHOTO_MODES):
        polled["before"] = readback_range(frame, polled["resources"])
        begin, end = polled["before"]
        if begin != end:
            if end - begin != settings["layout"]["pending_size"]:
                raise RuntimeError("expected one earlier replay readback on the held owner")
            polled["entry"] = [read(frame, begin), read(frame, begin + 8)]

    def finished(return_frame):
        polled["success"] = register(return_frame, "rax") & 0xFF == 1
        polled["status"] = read(return_frame, result, 4)
        if settings.get("mode") in ("replay", "tail", *PHOTO_MODES):
            polled["after"] = readback_range(return_frame, polled["resources"])
        record("readback_poll", index=draw["index"], **polled)

    at_return(frame, finished)


def prepare(frame, draw):
    stage = {"stream": argument(frame, 2)}
    draw["prepare"] = stage

    def finished(return_frame):
        stage["outcome"] = register(return_frame, "rax") & 0xFF
        record("scan_prepare", index=draw["index"], **stage)

    at_return(frame, finished)


def resource_operation(frame, draw, operation):
    if operation == "allocate":
        label, size, key = c_string(frame, argument(frame, 3)), argument(frame, 2), "allocations"
    else:
        label, size, key = c_string(frame, read(frame, argument(frame, 1) + 8)), argument(frame, 4), "uploads"
    item = {"label": label, "bytes": size, **owner(frame, argument(frame, 0))}
    draw[key].append(item)

    def finished(return_frame):
        item["success"] = register(return_frame, "rax") & 0xFF == 1
        record(operation, index=draw["index"], **item)

    at_return(frame, finished)


def replay_density(frame, draw):
    width, height = (argument(frame, index) & 0xFFFFFFFF for index in (1, 2))
    planes = argument(frame, 3)
    if (width, height) != (64, 32) or not planes or read(frame, planes + 24) or read(frame, planes + 32):
        raise RuntimeError("replay scanner is not using the admitted clear materialized density image")
    addresses = [read(frame, planes + offset) for offset in (0, 8, 16)]
    copy_density(frame, draw, addresses, width, height, "fj_observe_replay_density")


def copy_density(frame, draw, addresses, width, height, driver_symbol, *, role="density"):
    thread = frame.GetThread()
    process = thread.GetProcess()
    target = process.GetTarget()
    entry = target.FindSymbols(driver_symbol, lldb.eSymbolTypeCode)
    driver = str(target.GetExecutable())
    locations = {
        item.GetSymbol().GetStartAddress().GetLoadAddress(target)
        for item in entry
        if str(item.GetModule().GetFileSpec()) == driver
    }
    if len(locations) != 1 or not all(addresses):
        raise RuntimeError("replay density pointers or isolated driver observer cannot be bound")
    address = locations.pop()
    count = width * height
    operands = ", ".join(f"(void*){pointer}ull" for pointer in addresses)
    expression = (
        f"(unsigned long long)((void* (*)(void*, void*, void*, unsigned long long, void*, void*))0x{address:x})"
        f"({operands}, {count}ull, (void*){draw['stream']}ull, (void*){draw['context']}ull)"
    )
    options = lldb.SBExpressionOptions()
    options.SetIgnoreBreakpoints(True)
    options.SetUnwindOnError(True)
    options.SetLanguage(lldb.eLanguageTypeC)
    result = frame.EvaluateExpression(expression, options)
    if result.GetError().Fail() or not result.GetValueAsUnsigned():
        raise RuntimeError(f"driver could not copy the live scanner density planes: {result.GetError()}")
    status = lldb.SBError()
    location = result.GetValueAsUnsigned()
    data = process.ReadMemory(location, (count * 3 + 1) * 4, status)
    if status.Fail() or len(data) != (count * 3 + 1) * 4 or struct.unpack_from("<I", data)[0] != 0:
        raise RuntimeError("live scanner density plane copy was incomplete")
    receipt = Path(settings["receipt"])
    pathname = receipt.with_name(f"{receipt.stem}-{role}-{draw['index']}.f32")
    payload = data[4 : 4 + count * 4] if role == "glare" else data[4:]
    with pathname.open("xb") as output:
        output.write(payload)
    draw[role] = {
        "file": pathname.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "width": width,
        "height": height,
    }
    record(f"live_scanner_{role}", index=draw["index"], **draw[role])


def typed_field(value_type, name):
    fields = [field for field in value_type.fields if field.GetName() == name]
    if len(fields) != 1:
        raise RuntimeError(f"cannot bind {value_type.GetName()}::{name}")
    return fields[0]


def scan_payload(frame, params_name):
    params_type = frame.GetModule().FindFirstType(f"JuicerCuda::{params_name}PipelineRunParams")
    scan = typed_field(params_type, "scanStage")
    return argument(frame, 0) + scan.GetOffsetInBytes(), scan.GetType()


def read_cubic_grid(frame, params_name):
    nodes = ("Log2PchipXYZ", "PchipSlopeC", "PchipSlopeM", "PchipSlopeY")
    cells = ("PchipCellMin", "PchipCellMax")
    names = (*nodes, *cells)
    archived_offsets = settings["layout"].get("cubic_scan_stage")
    if archived_offsets is None:
        payload, payload_type = scan_payload(frame, params_name)
        grid_offset = typed_field(payload_type, "scanLutRes").GetOffsetInBytes()
        buffer_offsets = [typed_field(payload_type, f"scanLut{name}").GetOffsetInBytes() for name in names]
    else:
        stage_offset = archived_offsets[params_name.lower()]
        if type(stage_offset) is not int or stage_offset < 0 or stage_offset % 8 != 0:
            raise RuntimeError("archived scan stage offset must be an aligned byte offset")
        payload = argument(frame, 0) + stage_offset
        # The archived x64 ScanStagePayload starts with six pointers and the grid.
        grid_offset = 8 * len(names)
        buffer_offsets = range(0, grid_offset, 8)
    grid = read(frame, payload + grid_offset, 4)
    pointers = {name: read(frame, payload + offset) for name, offset in zip(names, buffer_offsets, strict=True)}
    if grid < 2 or not all(pointers.values()) or len(set(pointers.values())) != len(pointers):
        raise RuntimeError("cubic scanner did not bind six distinct buffers and a valid grid")
    return {
        "grid": grid,
        "node_dimensions": [grid] * 3,
        "cell_dimensions": [grid - 1] * 3,
        "pointers": pointers,
        "payload_bytes": 3 * 4 * (len(nodes) * grid**3 + len(cells) * (grid - 1) ** 3),
    }


def scanner_field(frame, name, field):
    compiled_layout = settings["layout"].get("scanner_payload")
    if compiled_layout:
        return compiled_layout[name][field]
    data_type = frame.GetModule().FindFirstType(f"JuicerCuda::{name}")
    member = typed_field(data_type, field)
    return member.GetOffsetInBytes(), member.GetType().GetByteSize()


def observe_materialized_density(frame, draw):
    params = argument(frame, 0)
    stage = draw["scan_medium"]
    printing = stage == "Print"
    params_type = f"{stage}PipelineRunParams"
    dimensions = (scanner_field(frame, params_type, field) for field in ("width", "height"))
    width, height = (read(frame, params + offset, size) for offset, size in dimensions)
    first_plane = 2 if settings["spectral_payload"] else 1
    addresses = [argument(frame, first_plane + lane) for lane in range(3)]
    geometry = tuple(settings.get("density_geometry", (4832, 2016)))
    if (width, height) != geometry or (not printing and argument(frame, first_plane + 6)) or len(set(addresses)) != 3:
        raise RuntimeError("scanner density observer requires the admitted image and three distinct clear planes")
    stage_offset, _ = scanner_field(frame, params_type, "scanStage")
    payload = params + stage_offset
    correction_offset, correction_size = scanner_field(frame, "ScanStagePayload", "correctionActive")
    glare_offset, glare_size = scanner_field(frame, "ScanStagePayload", "glarePercent")
    correction = read(frame, payload + correction_offset, correction_size)
    glare = read(frame, payload + glare_offset, glare_size)
    if correction or glare:
        raise RuntimeError("held scanner comparison requires correction and glare off")
    if settings.get("mode") == "print-photo":
        draw["correction_words"] = {"correctionActive": correction}
        for name in ("correctionSlope", "correctionOffset"):
            offset, size = scanner_field(frame, "ScanStagePayload", name)
            draw["correction_words"][name] = read(frame, payload + offset, size)
    color_offset, _ = scanner_field(frame, "ScanStagePayload", "scanColor")
    color_address = payload + color_offset
    names = ("cat02", "xyzToRgb", "outputGamutActive", "encoding")
    draw["scan_color"] = {}
    for name in names:
        field, size = scanner_field(frame, "ScanColorPayload", name)
        draw["scan_color"][name] = [read(frame, color_address + field + offset, 4) for offset in range(0, size, 4)]
    if settings["spectral_payload"] and not draw["scan_color"]["outputGamutActive"][0]:
        compiled_layout = settings["layout"].get("scanner_payload")
        size = (
            compiled_layout["ScannerSpectra"]["size"][1]
            if compiled_layout
            else frame.GetModule().FindFirstType("JuicerCuda::ScannerSpectra").GetByteSize()
        )
        error = lldb.SBError()
        spectra = frame.GetThread().GetProcess().ReadMemory(argument(frame, 1), size, error)
        if size != 2272 or error.Fail() or len(spectra) != size:
            raise RuntimeError("cannot capture the live spectral scanner packet")
        matrices = draw["scan_color"]["cat02"] + draw["scan_color"]["xyzToRgb"]
        packet = spectra + struct.pack("<18I", *matrices)
        name = f"scanner-packet-{draw['index']}.f32" if settings.get("mode") == "print-photo" else "scanner-packet.f32"
        filename = Path(settings["receipt"]).with_name(name)
        with filename.open("xb") as output:
            output.write(packet)
        draw["scanner_packet"] = {"file": filename.name, "sha256": hashlib.sha256(packet).hexdigest()}
    if settings.get("mode") == "print-photo":
        if not printing:
            raise RuntimeError("Print photograph reached a non-Print density launcher")
        draw["paper_planes"] = addresses
        symbol = "fj_observe_replay_density"
    else:
        symbol = "fj_observe_scanner_density"
    copy_density(frame, draw, addresses, width, height, symbol)


def observe_photo_glare(frame, draw):
    planes = argument(frame, 3 if settings["spectral_payload"] else 1)
    if not planes or [read(frame, planes + offset) for offset in (0, 8, 16)] != draw["paper_planes"]:
        raise RuntimeError("actual scanner kernel did not consume the observed paper planes")
    glare = read(frame, planes + 24)
    if bool(glare) != (settings["print_glare"] != "off") or "scanner_glare_pointer" in draw:
        raise RuntimeError("unexpected pre-white glare field at the Print scanner kernel")
    draw["scanner_glare_pointer"] = glare
    if glare:
        width, height = settings["density_geometry"]
        copy_density(frame, draw, [glare] * 3, width, height, "fj_observe_replay_density", role="glare")


def on_entry(frame, location, _):
    try:
        operation = entries[location.GetBreakpoint().GetID()]
        thread = frame.GetThread().GetThreadID()
        if operation == "render":
            render(frame, thread)
            return False
        draw = active.get(thread)
        if draw is None:
            return False
        if operation == "scanner_schedule":
            draw["launches"].append(settings["scanner_schedule"])
            record(operation, index=draw["index"], entry=settings["scanner_schedule"])
            if settings.get("cubic_grid"):
                draw["cubic_grid"] = read_cubic_grid(frame, draw["scan_medium"])
                record("cubic_grid", index=draw["index"], **draw["cubic_grid"])
            if settings.get("observe_density"):
                observe_materialized_density(frame, draw)
            if settings.get("mode") == "print-photo" and draw["index"] < 2:
                observe_materialized_density(frame, draw)
        elif operation == "photo_glare":
            if draw["index"] < 2:
                observe_photo_glare(frame, draw)
        elif operation == "spectral_launch":
            draw["launches"].append(operation)
            record(operation, index=draw["index"])
            if settings.get("mode") == "replay":
                replay_density(frame, draw)
        elif operation == "prepare":
            prepare(frame, draw)
        elif operation == "current_grain":
            if draw["current_grain"] is not None:
                raise RuntimeError("unexpected second Current grain launch")
            payload = argument(frame, 0)
            if read(frame, payload, 4) != 1:
                raise RuntimeError("Current grain launch payload is inactive")
            # GrainPayload seedBase, seedBaseNext, frameIndex and timeAlpha.
            draw["current_grain"] = [
                read(frame, payload + offset, size) for offset, size in ((24, 8), (32, 8), (72, 8), (80, 4))
            ]
            record("current_grain", index=draw["index"], words=draw["current_grain"])
        elif operation in ("allocate", "upload"):
            resource_operation(frame, draw, operation)
        elif operation == "retain":
            retain(frame, draw)
        elif operation == "poll":
            poll(frame, draw)
        elif "poll" in draw and "status" not in draw["poll"]:
            polled = draw["poll"]
            event = argument(frame, 0)
            if operation == "destroy_event":
                polled["destroyed"].append(event)
            else:
                predecessor = frames[draw["index"] - 1]["retained"] if draw["index"] else tail_predecessor

                def finished(return_frame):
                    status = register(return_frame, "rax") & 0xFFFFFFFF
                    flag = (
                        read(return_frame, predecessor["host"], 4)
                        if predecessor and event == predecessor["event"] and status == 0
                        else None
                    )
                    polled["queries"].append({"event": event, "status": status, "completed_flag": flag})

                at_return(frame, finished)
    except Exception as error:
        fail(error)
    return False


def bind_points(target, module, required):
    if not module.IsValid() or module.GetUUIDString() != settings["uuid"]:
        raise RuntimeError("scanner observer module or symbol identity differs")
    if settings.get("symbols") and Path(str(module.GetSymbolFileSpec())).resolve() != Path(settings["symbols"]):
        raise RuntimeError("scanner observer did not load the admitted symbol file")
    if set(settings["points"]) != set(required):
        raise RuntimeError("incomplete scanner producer, resource and terminal instrumentation")
    record(
        "debugger_module",
        module=str(module.GetFileSpec()),
        symbols=str(module.GetSymbolFileSpec()),
        uuid=module.GetUUIDString(),
    )
    text = module.FindSection(".text")
    for name, expected in required.items():
        declaration = settings["points"][name]
        address = module.ResolveFileAddress((text.GetFileAddress() if WINDOWS else 0) + declaration["offset"])
        names = {
            match.GetName()
            for match in (address.GetFunction(), address.GetSymbol())
            if match.IsValid() and match.GetStartAddress().GetFileAddress() == address.GetFileAddress()
        }
        if declaration["symbol"] not in names or expected not in declaration["symbol"]:
            raise RuntimeError(f"scanner observer could not verify the {name} symbol entry")
        bp = target.BreakpointCreateBySBAddress(address)
        if not bp.IsValid():
            raise RuntimeError(f"scanner observer could not bind {name}")
        bp.SetScriptCallbackFunction("observe_seed.on_entry")
        entries[bp.GetID()] = name
        record("symbol_entry", name=name, symbol=declaration["symbol"], file_address=address.GetFileAddress())


def install(debugger):
    global settings
    settings = json.loads(Path(os.environ["FJ_SCANNER_OBSERVER"]).read_text(encoding="utf-8"))
    target = debugger.GetSelectedTarget()
    module = target.AddModule(settings["module"], None, None, settings.get("symbols"))
    required = POINTS
    if settings.get("mode") == "replay":
        required = REPLAY_POINTS
    if settings.get("mode") == "print-photo":
        required = {
            **TAIL_POINTS,
            "scanner_schedule": "juicer_cuda_print_focused_scan_linear_density_rgb",
            "photo_glare": "scan_linear_density_rgb_kernel",
        }
        settings["scanner_schedule"] = required["scanner_schedule"]
    if settings.get("mode") == "current-photo":
        required = {**TAIL_POINTS, "current_grain": REPLAY_POINTS["current_grain"]}
    bind_points(target, module, required)


def stopped_predecessor(frame, ready):
    layout = settings["layout"]
    instance = ready["instance_data"]
    submission = instance + layout["owner_state"] + layout["instance_submission"]
    state = submission + layout["submission_state"] + layout["mutex_value"]
    optional_latch = state + layout["state_latch"]
    if read(frame, optional_latch) != 1:
        raise RuntimeError("measured request has no live Rust latch")
    latch = optional_latch + layout["latch_value"]
    key = latch + layout["latch_key"]
    bindings = [read(frame, key + offset) for offset in layout["key_bindings"]]
    identity = submission + layout["submission_identity"] + layout["identity_value"]
    instance_token = read(frame, identity + layout["instance_token"])
    latch_id = read(frame, latch + layout["latch_id"])
    next_id = read(frame, state + layout["state_next_id"])
    if (
        not instance_token
        or not latch_id
        or next_id != latch_id + 1
        or not all(bindings)
        or read(frame, key + layout["key_frame"]) != ready["token"]
        or read(frame, key + layout["key_device"], 4) != ready["device_ordinal"]
        or read(frame, key + layout["key_context"]) != ready["context"]
    ):
        raise RuntimeError("live Rust latch does not identify the measured request")
    runtime_arc = read(frame, instance + layout["owner_loaded"] + layout["admission_runtime"])
    if not runtime_arc:
        raise RuntimeError("measured instance has no live Rust Runtime")
    native = runtime_arc + layout["arc_value"] + layout["runtime_native"]
    if read(frame, native) != 1:
        raise RuntimeError("measured Runtime has no native owner")
    handle = read(frame, native + layout["native_value"] + layout["native_handle"])
    if not handle:
        raise RuntimeError("measured instance has no live native Runtime")
    owner_map = handle + layout["cuda_root"] + layout["root_context_map"]
    first = read(frame, owner_map + layout["map_first"])
    sentinel = read(frame, owner_map + layout["map_sentinel"]) if layout["map_sentinel"] is not None else 0
    count = read(frame, owner_map + layout["map_length"])
    if layout["map_first_next"]:
        first = read(frame, first + layout["node_next"])
    candidates = []
    visited = set()
    node = first
    for _ in range(count):
        if not node or node == sentinel or node in visited:
            raise RuntimeError("live Root context traversal ended before its recorded size")
        visited.add(node)
        payload = node + layout["node_value"]
        device = read(frame, payload + layout["map_key_device"], 4)
        context = read(frame, payload + layout["map_key_context"])
        epoch = read(frame, payload + layout["map_key_epoch"])
        resource = read(frame, payload + layout["map_frame_owner"])
        if device == ready["device_ordinal"] and context == ready["context"] and resource:
            state_view = owner(frame, resource)
            if (device, context, epoch) != tuple(state_view[name] for name in ("device", "context", "epoch")):
                raise RuntimeError("Root map entry and retained frame owner disagree")
            begin, end = readback_range(frame, resource)
            if begin != end:
                if end - begin != layout["pending_size"]:
                    raise RuntimeError("measured owner lacks a singleton retained scan readback")
                held = {
                    **state_view,
                    "host": read(frame, begin),
                    "event": read(frame, begin + 8),
                    "epoch_at_production": read(frame, begin + 16 + layout["identity_epoch"]),
                    "producer": [read(frame, begin + 16 + offset) for offset in layout["producer"]],
                    "route": read(frame, begin + 16 + layout["identity_route"], 4),
                }
                candidates.append(held)
        node = read(frame, node + layout["node_next"])
    if node != sentinel or len(candidates) != 1:
        raise RuntimeError("no unique measured Root owner retained the scan result")
    held = candidates[0]
    if (
        held["epoch_at_production"] != held["epoch"]
        or not held["host"]
        or not held["event"]
        or held["event"] == ready["stop_event"]
        or (held["producer"] and held["producer"][:3] != [latch_id, ready["token"], bindings[2]])
    ):
        raise RuntimeError("measured predecessor scan retention disagrees with its Rust latch")
    flag = read(frame, held["host"], 4)
    if flag != 0:
        raise RuntimeError("measured predecessor retained a nonzero scan flag")
    return {
        **held,
        "completed_flag": flag,
        "handle": handle,
        "instance": instance_token,
        "latch_id": latch_id,
        "next_id": next_id,
        "bindings": bindings,
        "stream": ready["stream"],
        "frame_token": ready["token"],
    }


def release_reporter(ready, action):
    path = Path(settings["release"])
    staged = path.with_suffix(".tmp")
    request = {key: ready[key] for key in ("run", "arm", "pid", "token")}
    staged.write_text(json.dumps({**request, "action": action}) + "\n", encoding="utf-8")
    staged.replace(path)


def tail(debugger):
    global settings, tail_predecessor
    settings = json.loads(Path(os.environ["FJ_SCANNER_OBSERVER"]).read_text(encoding="utf-8"))
    ready = None
    process = None
    status = None
    child_status = None
    debugger.SetAsync(False)
    child = subprocess.Popen(settings["command"])
    try:
        ready_path = Path(settings["ready"])
        while not ready_path.is_file():
            if child.poll() is not None:
                raise RuntimeError("measured child exited before its tail rendezvous")
            time.sleep(0.01)
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        expected_requests = settings["last_ordinal"] + 1
        if (
            ready["pid"] != child.pid
            or ready["run"] != settings["run"]
            or ready["arm"] != settings["arm"]
            or ready["ordinal"] != settings["last_ordinal"]
            or ready["token"] != ready["ordinal"]
            or ready["time_bits"] != struct.unpack("<Q", struct.pack("<d", float(ready["ordinal"])))[0]
            or ready["stop_status"] != 0
            or ready["render_status"] != 0
            or ready["callback_error"]
            or not ready["output_guard_valid"]
            or not ready["releases_valid"]
            or ready["source_releases"] != expected_requests
            or ready["destination_releases"] != expected_requests
        ):
            raise RuntimeError("measured child tail rendezvous failed admission")
        target = debugger.CreateTarget(settings["command"][0])
        listener = debugger.GetListener()
        attach_error = lldb.SBError()
        process = target.AttachToProcessWithID(listener, child.pid, attach_error)
        if attach_error.Fail() or process.GetState() != lldb.eStateStopped:
            raise RuntimeError(f"could not stop the measured child for late attachment: {attach_error}")
        thread = process.GetThreadByID(ready["render_thread"])
        if not thread.IsValid() or not thread.GetFrameAtIndex(0).IsValid():
            raise RuntimeError("reported render thread unavailable at the measured barrier")
        loaded = [target.GetModuleAtIndex(index) for index in range(target.GetNumModules())]
        loaded = [module for module in loaded if Path(str(module.GetFileSpec())).resolve() == Path(settings["module"])]
        if len(loaded) != 1:
            raise RuntimeError("measured module is missing or ambiguous in the attached process")
        module = loaded[0]
        if settings.get("symbols"):
            debugger.HandleCommand(f'target symbols add "{Path(settings["symbols"]).as_posix()}"')
        required = TAIL_POINTS
        if settings.get("scanner_schedule"):
            required = {**required, "scanner_schedule": settings["scanner_schedule"]}
        if settings.get("replay_tokens"):
            required = {**required, "current_grain": REPLAY_POINTS["current_grain"]}
        bind_points(target, module, required)
        tail_predecessor = stopped_predecessor(thread.GetFrameAtIndex(0), ready)
        record("measured_predecessor", **tail_predecessor)
        release_reporter(ready, "release")
        continue_error = process.Continue()
        if continue_error.Fail() or process.GetState() != lldb.eStateExited:
            raise RuntimeError(f"reporter child did not complete under LLDB: {continue_error}")
        status = process.GetExitStatus()
    except Exception as error:
        fail(error)
        if ready and not Path(settings["release"]).exists():
            release_reporter(ready, "abort")
        if process and process.IsValid() and process.GetState() == lldb.eStateStopped:
            process.Detach()
        if not ready and child.poll() is None:
            child.terminate()
    finally:
        child_status = child.wait()
        report = {
            "schema": "juicer-scanner-measured-tail-v1",
            "ready": ready,
            "predecessor": tail_predecessor,
            "frames": frames,
            "errors": errors,
            "exit": status,
            "child_exit": child_status,
            "pending_returns": len(returns),
        }
        Path(settings["receipt"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        passed = status == child_status == 0 and len(frames) == 1 and not errors and not active and not returns
        debugger.HandleCommand(f"quit {0 if passed else 1}")


def finish(debugger):
    process = debugger.GetSelectedTarget().GetProcess()
    status = process.GetExitStatus()
    exited = process.GetState() == lldb.eStateExited
    record("inferior_exit", status=status, exited=exited, pending_returns=len(returns))
    replay = settings.get("mode") == "replay"
    if settings.get("mode") in PHOTO_MODES:
        observed_draw = (
            ("density", "scanner_glare_pointer") if settings["mode"] == "print-photo" else ("current_grain",)
        )
        passed = (
            status == 0
            and exited
            and len(frames) == 3
            and not errors
            and not active
            and not returns
            and all(row["outcome"] == [0, 0, 0, 0] for row in frames)
            and all(all(row.get(field) is not None for field in observed_draw) for row in frames[:2])
        )
        report = {"status": "passed" if passed else "failed", "frames": frames, "errors": errors, "exit": status}
        Path(settings["receipt"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        debugger.HandleCommand(f"quit {0 if passed else 1}")
        return
    report = {
        "schema": "juicer-scanner-replay-observation-v2" if replay else "juicer-scanner-capacity-observation-v1",
        "frames": frames,
        "errors": errors,
        "inferior_exit": status,
        "exited": exited,
        "pending_returns": len(returns),
    }
    if replay:
        report["variant"] = settings["variant"]
        report["grain"] = settings["grain"]
        report["route"] = settings["route"]
        report["tokens"] = frames[0]["tokens"] if frames else None
    Path(settings["receipt"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    passed = (
        status == 0 and exited and len(frames) == (4 if replay else 2) and not errors and not active and not returns
    )
    debugger.HandleCommand(f"quit {0 if passed else 1}")
