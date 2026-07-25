$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python environment not found at $PythonExe. Follow the README installation steps first."
}

Set-Location -LiteralPath $ProjectRoot
& $PythonExe -m streamlit run "ui\app.py" `
    --server.address "127.0.0.1" `
    --server.port "8501" `
    --browser.gatherUsageStats "false"
exit $LASTEXITCODE

