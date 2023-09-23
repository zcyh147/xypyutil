param (
	[string]$text,
    [string]$filepath
)

# Add the System.Speech assembly
Add-Type -AssemblyName System.speech

# Create a SpeechSynthesizer object
$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer

# Set the output to the specified .wav file
$speak.SetOutputToWaveFile($filepath)

# Speak the text and save it to the .wav file
$speak.Speak($text)

# Dispose the SpeechSynthesizer object to release resources
$speak.Dispose()
