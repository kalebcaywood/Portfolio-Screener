# Creates a Quant Portfolio Lab shortcut on your Desktop.
# Right-click this file → "Run with PowerShell" to install.

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$batchFile  = Join-Path $projectDir "run.bat"
$desktop    = [Environment]::GetFolderPath("Desktop")
$shortcut   = Join-Path $desktop "Quant Portfolio Lab.lnk"

# Optional: install in Start Menu too
$startMenu  = [Environment]::GetFolderPath("Programs")
$startMenuShortcut = Join-Path $startMenu "Quant Portfolio Lab.lnk"

$WshShell = New-Object -comObject WScript.Shell

function Create-Shortcut($path) {
    $sc = $WshShell.CreateShortcut($path)
    $sc.TargetPath       = $batchFile
    $sc.WorkingDirectory = $projectDir
    $sc.IconLocation     = "$env:SystemRoot\System32\shell32.dll,137"   # chart icon
    $sc.WindowStyle      = 7                                            # minimized
    $sc.Description      = "Quantitative Portfolio Analytics — Streamlit app"
    $sc.Save()
    Write-Host "Created: $path"
}

Create-Shortcut $shortcut
Create-Shortcut $startMenuShortcut

Write-Host ""
Write-Host "Done. Double-click 'Quant Portfolio Lab' on your Desktop to launch."
Write-Host "It is also available in the Start Menu."
Write-Host ""
Write-Host "To auto-start on Windows login, copy the shortcut to:"
Write-Host "  shell:startup"
Write-Host "(paste that into the Run dialog — Win+R — and drop the shortcut there)"
Pause
