# Inno Setup installer (Windows)

1. Copy your plug-in bundle folder to `installer/Juicer.ofx.bundle/` so the layout is:

   - `installer/Juicer.iss`
   - `installer/Juicer.ofx.bundle/Contents/Win64/...`

2. Open `installer/Juicer.iss` in Inno Setup and click **Compile**.

The default install location is:

- `C:\Program Files\Common Files\OFX\Plugins\Juicer.ofx.bundle`

Uninstall removes that `Juicer.ofx.bundle` folder and nothing else.
