//! Current-module resource path. Native path units stay at this OS edge.

#[cfg(unix)]
use std::path::Component;
use std::path::{Path, PathBuf};

// `Path::parent` skips interior `.` components, while the C++ owner removes
// literal filename components before normalization. Keep the original OS units.
#[cfg(unix)]
fn parent(path: &Path) -> Option<PathBuf> {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    let units = path.as_os_str().as_bytes();
    let end = parent_end(units, 0, |unit| unit == b'/');
    (end != 0).then(|| PathBuf::from(OsStr::from_bytes(&units[..end])))
}

#[cfg(windows)]
fn is_separator(unit: u16) -> bool {
    unit == u16::from(b'/') || unit == u16::from(b'\\')
}

#[cfg(windows)]
fn root_name_end(units: &[u16]) -> usize {
    // Match MSVC filesystem's root-name parsing, including its device prefixes
    // and its server-only UNC root-name, rather than Rust's UNC prefix parsing.
    if units.len() >= 2
        && ((u16::from(b'A')..=u16::from(b'Z')).contains(&units[0])
            || (u16::from(b'a')..=u16::from(b'z')).contains(&units[0]))
        && units[1] == u16::from(b':')
    {
        2
    } else if units.len() >= 4
        && is_separator(units[0])
        && is_separator(units[3])
        && (units.len() == 4 || !is_separator(units[4]))
        && ((is_separator(units[1])
            && (units[2] == u16::from(b'?') || units[2] == u16::from(b'.')))
            || (units[1] == u16::from(b'?') && units[2] == u16::from(b'?')))
    {
        3
    } else if units.len() >= 3
        && is_separator(units[0])
        && is_separator(units[1])
        && !is_separator(units[2])
    {
        units[3..]
            .iter()
            .position(|&unit| is_separator(unit))
            .map_or(units.len(), |index| index + 3)
    } else {
        0
    }
}

#[cfg(windows)]
fn parent(path: &Path) -> Option<PathBuf> {
    use std::ffi::OsString;
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    let units: Vec<u16> = path.as_os_str().encode_wide().collect();
    let end = parent_end(&units, root_name_end(&units), is_separator);
    (end != 0).then(|| PathBuf::from(OsString::from_wide(&units[..end])))
}

fn parent_end<T: Copy>(units: &[T], root_name: usize, separator: impl Fn(T) -> bool) -> usize {
    let mut relative = root_name;
    while relative < units.len() && separator(units[relative]) {
        relative += 1;
    }
    let mut end = units.len();
    while end > relative && !separator(units[end - 1]) {
        end -= 1;
    }
    while end > relative && separator(units[end - 1]) {
        end -= 1;
    }
    end
}

fn resources(module: Option<&Path>) -> PathBuf {
    let Some(contents) = module.and_then(parent).and_then(|path| parent(&path)) else {
        return PathBuf::new();
    };
    #[cfg(unix)]
    {
        let mut normalized = PathBuf::new();
        for component in contents.join("Resources").components() {
            match component {
                Component::CurDir => {}
                Component::ParentDir => {
                    if normalized.file_name().is_some_and(|name| name != "..") {
                        normalized.pop();
                    } else if !normalized.has_root() {
                        normalized.push(component);
                    }
                }
                _ => normalized.push(component),
            }
        }
        normalized
    }
    #[cfg(windows)]
    {
        use std::ffi::OsString;
        use std::os::windows::ffi::{OsStrExt, OsStringExt};
        let mut units: Vec<u16> = contents.as_os_str().encode_wide().collect();
        let slash = u16::from(b'\\');
        let root_name = root_name_end(&units);
        // Apply the native append rules before parsing normalization's root:
        // a separator-only parent can acquire a UNC root-name from Resources.
        if root_name == units.len() {
            if root_name >= 3 {
                units.push(slash);
            }
        } else if !is_separator(units[units.len() - 1]) {
            units.push(slash);
        }
        units.extend("Resources".encode_utf16());
        let root_name = root_name_end(&units);
        let has_root_directory = units.get(root_name).is_some_and(|&unit| is_separator(unit));
        let dot = u16::from(b'.');
        let mut filenames: Vec<&[u16]> = Vec::new();
        // MSVC treats a UNC share (and device-path parts) as filenames. Rust's
        // component/push rules would instead protect them as a larger prefix.
        for filename in units[root_name..].split(|&unit| is_separator(unit)) {
            if filename.is_empty() || filename == [dot] {
                continue;
            }
            if filename == [dot, dot] {
                if filenames.last().is_some_and(|&name| name != [dot, dot]) {
                    filenames.pop();
                } else if !has_root_directory {
                    filenames.push(filename);
                }
            } else {
                filenames.push(filename);
            }
        }
        let mut normalized: Vec<u16> = units[..root_name]
            .iter()
            .map(|&unit| if is_separator(unit) { slash } else { unit })
            .collect();
        // The fixed final Resources name prevents an empty result or trailing
        // separator, including when it became part of a UNC root-name.
        if has_root_directory {
            normalized.push(slash);
        }
        for (index, filename) in filenames.iter().enumerate() {
            if index != 0 {
                normalized.push(slash);
            }
            normalized.extend_from_slice(filename);
        }
        PathBuf::from(OsString::from_wide(&normalized))
    }
}

pub(crate) fn resource_directory() -> PathBuf {
    resources(module_path().as_deref())
}

#[cfg(unix)]
pub(crate) fn clock_nanoseconds() -> Option<u64> {
    // libstdc++ high_resolution_clock aliases system_clock on the qualified
    // Linux target. Preserve its nanosecond epoch count and unsigned cast.
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(duration) => Some(duration.as_nanos() as u64),
        Err(error) => Some(0u64.wrapping_sub(error.duration().as_nanos() as u64)),
    }
}

#[cfg(any(windows, feature = "test-support"))]
pub(crate) fn qpc_nanoseconds(counter: i64, frequency: i64) -> Option<u64> {
    if frequency <= 0 {
        return None;
    }
    // Preserve the three qualified MSVC steady_clock integer paths, including
    // truncation before the final unsigned representation of signed nanoseconds.
    let ns = match frequency {
        10_000_000 => counter.wrapping_mul(100),
        24_000_000 => counter
            .wrapping_mul(41)
            .wrapping_add(counter.wrapping_mul(2) / 3),
        _ => (counter / frequency)
            .wrapping_mul(1_000_000_000)
            .wrapping_add((counter % frequency).wrapping_mul(1_000_000_000) / frequency),
    };
    Some(ns as u64)
}

#[cfg(windows)]
pub(crate) fn clock_nanoseconds() -> Option<u64> {
    let (mut counter, mut frequency) = (0i64, 0i64);
    // SAFETY: Both Win32 QPC functions fill their exact LARGE_INTEGER-sized
    // independent local outputs; calling convention is the vendored system ABI.
    let available = unsafe {
        QueryPerformanceFrequency(&mut frequency) != 0 && QueryPerformanceCounter(&mut counter) != 0
    };
    if !available {
        return None;
    }
    qpc_nanoseconds(counter, frequency)
}

#[cfg(unix)]
#[repr(C)]
struct DlInfo {
    filename: *const std::ffi::c_char,
    base: *mut std::ffi::c_void,
    symbol: *const std::ffi::c_char,
    address: *mut std::ffi::c_void,
}

#[cfg(unix)]
#[link(name = "dl")]
unsafe extern "C" {
    fn dladdr(address: *const std::ffi::c_void, info: *mut DlInfo) -> std::ffi::c_int;
}

#[cfg(unix)]
fn module_path() -> Option<PathBuf> {
    use std::ffi::{CStr, OsStr};
    use std::os::unix::ffi::OsStrExt;
    let mut info = DlInfo {
        filename: std::ptr::null(),
        base: std::ptr::null_mut(),
        symbol: std::ptr::null(),
        address: std::ptr::null_mut(),
    };
    // SAFETY: This function is in the actual containing module; dladdr fills
    // local storage. Its filename remains valid while that module is loaded.
    if unsafe { dladdr((resource_directory as *const ()).cast(), &mut info) } == 0
        || info.filename.is_null()
    {
        return None;
    }
    // SAFETY: Successful dladdr supplies a module-lived NUL-terminated string.
    let bytes = unsafe { CStr::from_ptr(info.filename) }.to_bytes();
    Some(PathBuf::from(OsStr::from_bytes(bytes)))
}

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn QueryPerformanceCounter(counter: *mut i64) -> i32;
    fn QueryPerformanceFrequency(frequency: *mut i64) -> i32;
    fn GetModuleHandleExW(
        flags: u32,
        address: *const u16,
        module: *mut *mut std::ffi::c_void,
    ) -> i32;
    fn GetModuleFileNameW(module: *mut std::ffi::c_void, filename: *mut u16, size: u32) -> u32;
    fn SetLastError(error: u32);
    fn GetLastError() -> u32;
}

#[cfg(windows)]
fn module_path() -> Option<PathBuf> {
    module_path_with_capacity(260)
}

#[cfg(windows)]
fn module_path_with_capacity(capacity: usize) -> Option<PathBuf> {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;
    let mut module = std::ptr::null_mut();
    // SAFETY: FROM_ADDRESS interprets this code address rather than reading a
    // string; UNCHANGED_REFCOUNT leaves ownership with the loaded binary.
    if unsafe { GetModuleHandleExW(0x4 | 0x2, resource_directory as *const u16, &mut module) } == 0
    {
        return None;
    }
    let mut buffer = vec![0; capacity];
    loop {
        let capacity = u32::try_from(buffer.len()).ok()?;
        // SAFETY: Module is binary-lived, buffer is writable for capacity UTF-16
        // units, and last-error access is local to this calling thread.
        let (length, error) = unsafe {
            SetLastError(0);
            let length = GetModuleFileNameW(module, buffer.as_mut_ptr(), capacity);
            (length as usize, GetLastError())
        };
        if length == 0 {
            return None;
        }
        if length < buffer.len() || error != 122 {
            buffer.truncate(length);
            return Some(PathBuf::from(OsString::from_wide(&buffer)));
        }
        buffer.resize(buffer.len().checked_mul(2)?, 0);
    }
}

#[cfg(feature = "test-support")]
pub(crate) fn characterize(mode: u32, path: &Path) -> PathBuf {
    match mode {
        0 => resource_directory(),
        1 => resources(Some(path)),
        #[cfg(windows)]
        3 => resources(module_path_with_capacity(1).as_deref()),
        _ => resources(None),
    }
}

#[cfg(all(feature = "test-support", unix))]
pub(crate) fn abi() -> [usize; 6] {
    [
        std::mem::size_of::<DlInfo>(),
        std::mem::align_of::<DlInfo>(),
        std::mem::offset_of!(DlInfo, filename),
        std::mem::offset_of!(DlInfo, base),
        std::mem::offset_of!(DlInfo, symbol),
        std::mem::offset_of!(DlInfo, address),
    ]
}

#[cfg(all(feature = "test-support", windows))]
pub(crate) fn abi() -> [usize; 8] {
    [
        std::mem::size_of::<u16>(),
        std::mem::size_of::<u32>(),
        std::mem::size_of::<i32>(),
        std::mem::size_of::<*mut std::ffi::c_void>(),
        std::mem::align_of::<*mut std::ffi::c_void>(),
        0x4 | 0x2,
        std::mem::size_of::<i64>(),
        std::mem::align_of::<i64>(),
    ]
}
