-- Resolve scripting API companion to capture.py. All changes target a fresh
-- imported copy. The original project is exported by name and never saved.
local resolve = assert(bmd.scriptapp('Resolve'), 'Resolve unavailable')
local manager = resolve:GetProjectManager()
local directory = assert(os.getenv('JUICER_RESOLVE_CAPTURE_DIR'))
local original = assert(os.getenv('JUICER_RESOLVE_ORIGINAL_PROJECT'))
local name = assert(os.getenv('JUICER_RESOLVE_CAPTURE_PROJECT'))
local action = assert(os.getenv('JUICER_RESOLVE_CAPTURE_ACTION'))
assert(name ~= original, 'capture project must be a copy')
local report_path = directory .. '/' .. action .. '.txt'
local existing_report = io.open(report_path, 'r')
if existing_report then existing_report:close(); error('capture report already exists') end
local report = assert(io.open(report_path, 'w'))
local function record(key, value)
    report:write(key .. '=' .. tostring(value) .. '\n')
    report:flush()
end
local function properties(prefix, values)
    for key, value in pairs(values or {}) do record(prefix .. '.' .. tostring(key), value) end
end
local function close_current()
    local current = manager:GetCurrentProject()
    if current and not manager:CloseProject(current) then
        -- Resolve exposes an empty starter project at the project manager.
        assert(current:GetName() == 'Untitled Project' and current:GetTimelineCount() == 0,
            'close without saving failed for a populated project')
    end
end
if action == 'copy' then
    for _, existing in pairs(manager:GetProjectListInCurrentFolder()) do
        assert(existing ~= name, 'capture copy already exists')
    end
    close_current()
    local archive = directory .. '/original.drp'
    assert(manager:ExportProject(original, archive, false), 'original export failed')
    assert(manager:ImportProject(archive, name), 'copy import failed')
    local project = assert(manager:LoadProject(name), 'copy load failed')
    assert(project:GetName() == name)
    record('project', project:GetName())
elseif action == 'load' then
    close_current()
    local project = assert(manager:LoadProject(name), 'copy load failed')
    assert(project:GetName() == name)
    record('project', project:GetName())
elseif action == 'close' then
    local project = assert(manager:GetCurrentProject())
    assert(project:GetName() == name, 'refusing to close another project')
    assert(manager:CloseProject(project), 'close copy without saving failed')
    record('closed_copy', true)
    resolve:Quit()
elseif action == 'inspect' or action == 'export' then
    local project = assert(manager:GetCurrentProject())
    assert(project:GetName() == name, 'refusing to operate on another project')
    local timeline = assert(project:GetCurrentTimeline())
    record('project', project:GetName())
    record('timeline', timeline:GetName())
    record('timecode', timeline:GetCurrentTimecode())
    properties('project.setting', project:GetSetting())
    local clips = assert(timeline:GetItemListInTrack('video', 1))
    assert(#clips >= 2, 'clip 2 unavailable')
    for index, clip in ipairs(clips) do
        record('clip.' .. index .. '.name', clip:GetName())
        record('clip.' .. index .. '.start', clip:GetStart())
        record('clip.' .. index .. '.end', clip:GetEnd())
        properties('clip.' .. index, clip:GetProperty())
    end
    local frame = clips[2]:GetStart()
    local rate = tonumber(timeline:GetSetting('timelineFrameRate'))
    assert(rate == 24, 'inspect timecode conversion for non-24fps timeline')
    local seconds = math.floor(frame / rate)
    local timecode = string.format('%02d:%02d:%02d:%02d', math.floor(seconds/3600), math.floor(seconds/60)%60, seconds%60, frame%rate)
    assert(timeline:SetCurrentTimecode(timecode))
    record('selected_timecode', timecode)
    if action == 'export' then
        assert(project:ExportCurrentFrameAsStill(directory .. '/observed-clip-2.tif'), 'frame export failed')
        record('exported', true)
    end
else
    error('unsupported capture action')
end
report:close()
