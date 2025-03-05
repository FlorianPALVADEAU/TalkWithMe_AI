    export const checkUserAccesses = async (navigator: Navigator): Promise<boolean> => {
        let hasAccess = false

        if (!('webkitSpeechRecognition' in window)) {
            console.error('Speech recognition is not supported in this browser.');
            return false;
        }

        navigator.mediaDevices.enumerateDevices().then(devices => {
            console.log(devices);
            const audioInputDevices = devices.filter(device => device.kind === 'audioinput');
            if (audioInputDevices.length === 0) {
                console.error('No audio input devices found');
                return false;
            }
            // verify if the user has a microphone
            const hasMicrophone = audioInputDevices.length > 0;
            if (!hasMicrophone) {
                console.error('No microphone found');
                return false;
            }
        });
        
        
        try {
            const get = await navigator.mediaDevices.getUserMedia({ audio: true })

            if (get) {
                console.log('Microphone permissions granted')
                hasAccess = true
            } else {
                console.error('Error while granting microphone permissions')
                hasAccess = false
            }
        
        } catch (error) {
            console.error('Error while starting voice recognition:', error)
            hasAccess = false
        }
        
        return hasAccess
    }

    export default checkUserAccesses