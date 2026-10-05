$levels = @(1, 2, 4, 8)

$allResults = @()

foreach ($concurrency in $levels) {

    Write-Host ""
    Write-Host "====================================="
    Write-Host "Concurrency: $concurrency"
    Write-Host "====================================="

    $testStart = Get-Date

    $jobs = 1..$concurrency | ForEach-Object {

        $id = ($concurrency * 1000) + $_

        Start-Job -ScriptBlock {

            param($id)

            $body = @{
                request_id = $id
                prompt = "Explain machine learning in one sentence."
                max_tokens = 32
                temperature = 0
            } | ConvertTo-Json

            $clientStart = Get-Date

            try {

                $response = Invoke-RestMethod `
                    -Method POST `
                    -Uri "http://127.0.0.1:3000/v1/inference" `
                    -ContentType "application/json" `
                    -Body $body

                $clientElapsed = ((Get-Date) - $clientStart).TotalMilliseconds

                [PSCustomObject]@{
                    Request = $id
                    Success = $true

                    ClientMs =
                        [math]::Round($clientElapsed, 2)

                    QueueMs =
                        [math]::Round(
                            $response.timing.queue_ms,
                            2
                        )

                    ModelTTFT =
                        [math]::Round(
                            $response.timing.model_ttft_ms,
                            2
                        )

                    E2ETTFT =
                        [math]::Round(
                            $response.timing.end_to_end_ttft_ms,
                            2
                        )

                    TPOT =
                        [math]::Round(
                            $response.timing.tpot_ms,
                            2
                        )

                    TokensSec =
                        [math]::Round(
                            $response.timing.tokens_per_second,
                            2
                        )

                    Generated =
                        $response.usage.generated_tokens

                    TotalMs =
                        [math]::Round(
                            $response.timing.total_ms,
                            2
                        )
                }

            }
            catch {

                [PSCustomObject]@{
                    Request = $id
                    Success = $false
                    Error = $_.Exception.Message
                }

            }

        } -ArgumentList $id
    }

    $jobs | Wait-Job | Out-Null

    $results =
        $jobs | Receive-Job

    $jobs | Remove-Job

    $testElapsed =
        ((Get-Date) - $testStart).TotalSeconds

    $results |
        Format-Table `
            Request,
            Success,
            ClientMs,
            QueueMs,
            ModelTTFT,
            E2ETTFT,
            TPOT,
            TokensSec,
            Generated,
            TotalMs `
            -AutoSize

    $successful =
        $results |
        Where-Object { $_.Success }

    if ($successful.Count -gt 0) {

        $generatedTotal =
            (
                $successful |
                Measure-Object Generated -Sum
            ).Sum

        $systemThroughput =
            $generatedTotal / $testElapsed

        Write-Host ""
        Write-Host "Wall time: $([math]::Round($testElapsed,2)) sec"
        Write-Host "Generated tokens: $generatedTotal"
        Write-Host "System throughput: $([math]::Round($systemThroughput,2)) tokens/sec"

    }

    foreach ($result in $results) {

        $result |
            Add-Member `
                -NotePropertyName Concurrency `
                -NotePropertyValue $concurrency

        $allResults += $result
    }
}

Write-Host ""
Write-Host "====================================="
Write-Host "FINAL SUMMARY"
Write-Host "====================================="

$allResults |
    Format-Table `
        Concurrency,
        Request,
        ClientMs,
        QueueMs,
        ModelTTFT,
        E2ETTFT,
        TPOT,
        TokensSec,
        Generated `
        -AutoSize

Write-Host ""
Write-Host "Final server metrics:"

Invoke-RestMethod `
    http://127.0.0.1:3000/metrics |
    ConvertTo-Json
