$exclude = @('algs', 'arches', 'contours', 'denoise')

$dev = Get-ChildItem -Recurse -File -Path 'D:\anvo\yaaat-dev' |
    Where-Object { $rel = $_.FullName.Replace('D:\anvo\yaaat-dev\', ''); -not ($exclude | Where-Object { $rel.StartsWith($_) }) } |
    Get-FileHash -Algorithm MD5 |
    Select-Object @{N='File';E={Split-Path $_.Path -Leaf}}, Hash

$temp = Get-ChildItem -Recurse -File -Path 'D:\anvo\yaaat-temp' |
    Where-Object { $rel = $_.FullName.Replace('D:\anvo\yaaat-temp\', ''); -not ($exclude | Where-Object { $rel.StartsWith($_) }) } |
    Get-FileHash -Algorithm MD5 |
    Select-Object @{N='File';E={Split-Path $_.Path -Leaf}}, Hash

$all = ($dev.File + $temp.File) | Sort-Object -Unique
$all | ForEach-Object {
    $f = $_
    $l = ($dev | Where-Object File -eq $f).Hash
    $r = ($temp | Where-Object File -eq $f).Hash
    $match = if ($l -and $r) { $l.Trim().ToUpper() -eq $r.Trim().ToUpper() } else { 'MISSING' }
    [PSCustomObject]@{File=$f; dev=$l; temp=$r; Match=$match}
} | Format-Table -AutoSize | Out-String | Out-File D:\anvo\yaaat-dev\compare_excl_output.txt