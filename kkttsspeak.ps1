param([string]$text)
Add-Type -TypeDefinition @'
    using System.Runtime.InteropServices;
    [Guid("96749377-3391-11D2-9EE3-00C04F797396"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface ISpObjectToken { }
    [Guid("5B559F40-E952-11D2-BB91-00C04F8EE6C0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface ISpVoice { }
    [Guid("5B559F41-E952-11D2-BB91-00C04F8EE6C0"), ClassInterface(ClassInterfaceType.None), ComImport]
    class SpVoiceClass { }
'@
$voice = New-Object ISpVoice -ComObject SAPI.SpVoice
$voice.Speak($text)
