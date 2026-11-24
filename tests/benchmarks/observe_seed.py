"""Observe a natural scanner capacity run in its symbol-matched LLDB target."""

import json
import os
from pathlib import Path

import lldb


WINDOWS = os.name == "nt"
POINTS = {
    "render": "fj_cuda_render",
    "spectral_launch": "scan_linear_density_rgb_kernel",
    "allocate": "allocate_owned_device",
    "upload": "enqueue_host_to_device_copy",
    "retain": "retain_scan_error_readback",
    "poll": "poll_scan_error_readbacks",
    "query": "cudaEventQuery",
    "destroy_event": "cudaEventDestroy",
}
settings = {}
entries = {}
returns = {}
frames = []
active = {}
errors = []


def record(event, **fields):
    with Path(settings["trace"]).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, **fields}, sort_keys=True) + "\n")


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
        returns.pop(location.GetBreakpoint().GetID())(frame)
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
    draw["retained"] = held

    def finished(return_frame):
        held["success"] = register(return_frame, "rax") & 0xFF == 1
        held["handoff"] = read(return_frame, host_ref) == 0 and read(return_frame, event_ref) == 0
        record("readback_retained", index=draw["index"], **held)

    at_return(frame, finished)


def poll(frame, draw):
    if "poll" in draw:
        raise RuntimeError("more than one scan poll for a capacity render")
    polled = {**owner(frame, argument(frame, 0)), "stream": argument(frame, 1), "queries": [], "destroyed": []}
    draw["poll"] = polled
    result = argument(frame, 2)

    def finished(return_frame):
        polled["success"] = register(return_frame, "rax") & 0xFF == 1
        polled["status"] = read(return_frame, result, 4)
        record("readback_poll", index=draw["index"], **polled)

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
        if operation == "spectral_launch":
            draw["launches"].append(operation)
            record(operation, index=draw["index"])
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
                predecessor = frames[draw["index"] - 1]["retained"] if draw["index"] else None

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


def install(debugger):
    global settings
    settings = json.loads(Path(os.environ["FJ_SCANNER_OBSERVER"]).read_text(encoding="utf-8"))
    target = debugger.GetSelectedTarget()
    module = target.AddModule(settings["module"], None, None, settings.get("symbols"))
    if not module.IsValid() or module.GetUUIDString() != settings["uuid"]:
        raise RuntimeError("scanner observer module or symbol identity differs")
    if set(settings["points"]) != set(POINTS):
        raise RuntimeError("incomplete scanner producer, resource and terminal instrumentation")
    record(
        "debugger_module",
        module=str(module.GetFileSpec()),
        symbols=str(module.GetSymbolFileSpec()),
        uuid=module.GetUUIDString(),
    )
    text = module.FindSection(".text")
    for name, expected in POINTS.items():
        declaration = settings["points"][name]
        address = module.ResolveFileAddress((text.GetFileAddress() if WINDOWS else 0) + declaration["offset"])
        symbol = address.GetSymbol()
        if (
            not symbol.IsValid()
            or symbol.GetStartAddress().GetFileAddress() != address.GetFileAddress()
            or declaration["symbol"] != symbol.GetName()
            or expected not in declaration["symbol"]
        ):
            raise RuntimeError(f"scanner observer could not verify the {name} symbol entry")
        bp = target.BreakpointCreateBySBAddress(address)
        if not bp.IsValid():
            raise RuntimeError(f"scanner observer could not bind {name}")
        bp.SetScriptCallbackFunction("observe_seed.on_entry")
        entries[bp.GetID()] = name
        record("symbol_entry", name=name, symbol=symbol.GetName(), file_address=address.GetFileAddress())


def finish(debugger):
    process = debugger.GetSelectedTarget().GetProcess()
    status = process.GetExitStatus()
    exited = process.GetState() == lldb.eStateExited
    record("inferior_exit", status=status, exited=exited, pending_returns=len(returns))
    report = {
        "schema": "juicer-scanner-capacity-observation-v1",
        "frames": frames,
        "errors": errors,
        "inferior_exit": status,
        "exited": exited,
        "pending_returns": len(returns),
    }
    Path(settings["receipt"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    passed = status == 0 and exited and len(frames) == 2 and not errors and not active and not returns
    debugger.HandleCommand(f"quit {0 if passed else 1}")
