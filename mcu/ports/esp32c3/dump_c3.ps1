# Chunked flash backup for the ESP32-C3 on COM7.
# Partition table (read separately) says: bootloader+nvs+phy_init occupy
# 0x0-0x10000 and the factory app occupies 0x10000-0x310000. There is no
# SPIFFS on this board. Only regions that exist are dumped.
# 256 KB chunks with retries, then concatenate and hash.
$ErrorActionPreference = "Continue"
$py  = "$env:USERPROFILE\.espressif\python_env\idf6.0_py3.11_env\Scripts\python.exe"
$out = "C:\esp\board_backup_c3_20260822"
New-Item -ItemType Directory -Force -Path "$out\chunks" | Out-Null

$regions = @(
    @{ name = "boot"; start = 0x0;     size = 0x10000  },
    @{ name = "app";  start = 0x10000; size = 0x300000 }
)
$chunk = 0x40000   # 256 KB

foreach ($r in $regions) {
    $parts = @()
    for ($off = 0; $off -lt $r.size; $off += $chunk) {
        $len = [Math]::Min($chunk, $r.size - $off)
        $addr = $r.start + $off
        $f = "{0}\chunks\{1}_{2:x8}.bin" -f $out, $r.name, $addr
        $parts += $f
        if ((Test-Path $f) -and ((Get-Item $f).Length -eq $len)) {
            Write-Host ("SKIP {0} 0x{1:x} ({2} bytes, already have it)" -f $r.name, $addr, $len)
            continue
        }
        $ok = $false
        for ($try = 1; $try -le 5 -and -not $ok; $try++) {
            Write-Host ("READ {0} 0x{1:x} len 0x{2:x} attempt {3}" -f $r.name, $addr, $len, $try)
            & $py -m esptool --chip esp32c3 --port COM7 --baud 460800 --no-stub `
                read-flash $addr $len $f 2>&1 | Select-Object -Last 2
            if ((Test-Path $f) -and ((Get-Item $f).Length -eq $len)) { $ok = $true }
            else { Start-Sleep -Seconds 3 }
        }
        if (-not $ok) { Write-Host ("FAILED {0} 0x{1:x} -- BACKUP INCOMPLETE" -f $r.name, $addr); exit 1 }
    }
    $dst = "$out\c3_$($r.name).bin"
    $fs = [System.IO.File]::Create($dst)
    foreach ($p in $parts) {
        $b = [System.IO.File]::ReadAllBytes($p)
        $fs.Write($b, 0, $b.Length)
    }
    $fs.Close()
    $h = (Get-FileHash $dst -Algorithm SHA256).Hash
    Write-Host ("BACKUP {0}: {1} bytes sha256={2}" -f $r.name, (Get-Item $dst).Length, $h)
}
Copy-Item C:\esp\bk_c3\parttable.bin "$out\c3_parttable.bin" -Force
$h = (Get-FileHash "$out\c3_parttable.bin" -Algorithm SHA256).Hash
Write-Host ("BACKUP parttable: {0} bytes sha256={1}" -f (Get-Item "$out\c3_parttable.bin").Length, $h)
Write-Host "BACKUP COMPLETE"
