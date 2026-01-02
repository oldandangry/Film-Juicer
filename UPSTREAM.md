# agx-emulsion Reference

This project mirrors the modelling work from [andreavolpato/agx-emulsion](https://github.com/andreavolpato/agx-emulsion). Instead of shipping snapshots of that repository inside the plug-in bundle, fetch the upstream sources directly when you need to inspect modelling details:

```bash
# from the repo root
git clone https://github.com/andreavolpato/agx-emulsion.git external/agx-emulsion
```

Keep the clone outside the plug-in artefacts so Resolve deployments stay lean. For parity questions, note the commit hash used by the port in issue trackers or release notes.
