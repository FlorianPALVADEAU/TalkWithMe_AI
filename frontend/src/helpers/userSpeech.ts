/* eslint-disable @typescript-eslint/no-explicit-any */

let recognition: any = null;

export const initializeRecognition = () => {
    recognition = new (window as any).webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'fr-FR';

    recognition.onresult = (event: any) => {
        const result = event.results[event.results.length - 1];
        const text = result[0].transcript;
        console.log('Recognized text:', text);
    };

    recognition.onstart = () => {
        console.log('Recognition started');
    };

    recognition.onerror = (event: any) => {
        console.error('Recognition error:', event.error);
        if (event.error === 'not-allowed') {
            console.error('User denied microphone access');
        } else if (event.error === 'network') {
            console.error('Network issue while starting recognition');
            alert('Network issue: please check your internet connection and try again.');
            // setTimeout(() => {
            //     recognition.start();
            // }, 5000);
        }
    };
    

    recognition.onspeechend = () => {
        setTimeout(() => {
            stopRecognition();
        }, 2000);
    };
};

export const startRecognition = () => {
    if (recognition) {
        recognition.start();
    }
};

export const stopRecognition = () => {
    if (recognition) {
        console.log('Speech ended');
        recognition.stop();
    }
};
