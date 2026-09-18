param ([string]$ImagePath)

try {
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    $asTaskGeneric = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.ContainsGenericParameters } | Select-Object -First 1

    function AwaitWinRT($asyncOp, [Type]$type) {
        $asTask = $asTaskGeneric.MakeGenericMethod($type)
        $netTask = $asTask.Invoke($null, @($asyncOp))
        $netTask.Wait(-1) | Out-Null
        return $netTask.Result
    }

    [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
    [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime] | Out-Null
    [Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType = WindowsRuntime] | Out-Null

    $abs = (Resolve-Path $ImagePath).Path
    $fileOp = [Windows.Storage.StorageFile]::GetFileFromPathAsync($abs)
    $file = AwaitWinRT $fileOp ([Windows.Storage.StorageFile])

    $streamOp = $file.OpenAsync([Windows.Storage.FileAccessMode]::Read)
    $stream = AwaitWinRT $streamOp ([Windows.Storage.Streams.IRandomAccessStream])

    $decoderOp = [Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)
    $decoder = AwaitWinRT $decoderOp ([Windows.Graphics.Imaging.BitmapDecoder])

    $bmpOp = $decoder.GetSoftwareBitmapAsync()
    $bmp = AwaitWinRT $bmpOp ([Windows.Graphics.Imaging.SoftwareBitmap])

    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
    if ($null -eq $engine) {
        $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages[0])
    }

    $ocrOp = $engine.RecognizeAsync($bmp)
    $res = AwaitWinRT $ocrOp ([Windows.Media.Ocr.OcrResult])

    $words = @()
    foreach ($line in $res.Lines) {
        foreach ($w in $line.Words) {
            $words += [PSCustomObject]@{
                text = $w.Text
                x = [Math]::Round($w.BoundingRect.X, 1)
                y = [Math]::Round($w.BoundingRect.Y, 1)
                w = [Math]::Round($w.BoundingRect.Width, 1)
                h = [Math]::Round($w.BoundingRect.Height, 1)
            }
        }
    }
    $words | ConvertTo-Json -Compress
}
catch {
    Write-Output "[]"
}
