param(
    [string]$Executable = '.\test.exe',
    [ValidateRange(1,1000000000)][long]$N = 1048576,
    [ValidateRange(1,128)][int]$Ranks = 4,
    [ValidateRange(1,900)][int]$Repetitions = 100,
    [ValidateRange(1,60)][int]$LaunchDelaySeconds = 5
)
$ErrorActionPreference = 'Stop'
$executablePath = (Resolve-Path -LiteralPath $Executable).Path
$processes = @()
$commonStart = [DateTimeOffset]::UtcNow.AddSeconds($LaunchDelaySeconds).ToUnixTimeMilliseconds()
try {
    for ($rank = 0; $rank -lt $Ranks; $rank++) {
        $start = New-Object System.Diagnostics.ProcessStartInfo
        $start.FileName = $executablePath
        $start.Arguments = "$N $Ranks $Repetitions $rank $commonStart"
        $start.UseShellExecute = $false
        $start.CreateNoWindow = $true
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        $process = [System.Diagnostics.Process]::Start($start)
        $processes += $process
    }
    $failed = $false
    foreach ($process in $processes) {
        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        Write-Output $stdout.TrimEnd()
        if ($stderr) { Write-Output $stderr.TrimEnd() }
        if ($process.ExitCode -ne 0) { $failed = $true }
    }
    if ($failed) { exit 1 }
} finally {
    foreach ($process in $processes) { $process.Dispose() }
}
