# Allow the OutreachLM training node (laptop 2) to reach the monitor on this
# machine. Windows Firewall blocks unsolicited inbound connections by default,
# so without this rule laptop 2's telemetry will silently fail to connect --
# training will continue (by design) but you will see nothing.
#
# RUN THIS AS ADMINISTRATOR:
#   Right-click PowerShell -> "Run as administrator", then:
#   cd C:\Users\josep\Desktop\OutreachLM
#   .\scripts\allow_monitor_port.ps1

param(
    [int]$Port = 51799,
    [string]$RuleName = "OutreachLM Training Monitor"
)

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This script must be run as Administrator."
    exit 1
}

$existing = Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Rule '$RuleName' already exists. Removing and recreating..."
    Remove-NetFirewallRule -DisplayName $RuleName
}

# Scoped to Private networks only. Do not widen this to Public: it would expose
# the collector port on untrusted networks such as cafe or airport WiFi.
New-NetFirewallRule `
    -DisplayName $RuleName `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort $Port `
    -Profile Private `
    -Description "Inbound telemetry from the OutreachLM training node." | Out-Null

Write-Host "Created inbound TCP allow rule on port $Port (Private profile only)."
Write-Host ""
Write-Host "This machine's addresses:"
Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
    ForEach-Object { "  {0}:{1}  ({2})" -f $_.IPAddress, $Port, $_.InterfaceAlias }
