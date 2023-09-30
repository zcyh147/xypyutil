# kkttsspeak.ps1 'Hello there'
# kkttsspeak.ps1 -message 'Hello there'
[cmdletbinding()]
param(
    [Parameter(Position = 1, Mandatory = $true)]
    [String]
    $message
)
Add-Type -AssemblyName System.Speech
$synth = New-Object -TypeName System.Speech.Synthesis.SpeechSynthesizer
$synth.Speak($message)
