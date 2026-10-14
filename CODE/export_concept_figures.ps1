# Export the three editable SVG concepts as vector PDFs with Inkscape.
$ErrorActionPreference = 'Stop'
$figureRoot = Join-Path (Split-Path $PSScriptRoot -Parent) 'paper/figs'
$inkscapeCommand = Get-Command 'inkscape.com' -ErrorAction SilentlyContinue
if ($inkscapeCommand) {
    $inkscapeExe = $inkscapeCommand.Source
} else {
    $inkscapeExe = Join-Path $env:ProgramFiles 'Inkscape/bin/inkscape.com'
}
if (-not (Test-Path -LiteralPath $inkscapeExe)) {
    throw 'Inkscape was not found. Install it or add inkscape.com to PATH.'
}
foreach ($stem in @('fig1_study_overview', 'fig5_qtransport_workflow', 'fig7_information_staircase')) {
    $sourcePath = Join-Path $figureRoot "$stem.svg"
    $outputPath = Join-Path $figureRoot "$stem.pdf"
    & $inkscapeExe $sourcePath '--export-type=pdf' '--export-area-drawing' "--export-filename=$outputPath" 2>$null
    if ($LASTEXITCODE -ne 0) { throw "Inkscape export failed: $stem" }
    Write-Output "$stem.pdf"
}
