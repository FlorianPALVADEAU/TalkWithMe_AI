import { useEffect, useLayoutEffect, useState } from 'react';
import './styles/App.css';
import checkUserAccesses from './helpers/checkUserAccesses';
import { ReactMic } from 'react-mic';
import { postUserSpeech } from './api/apiCalls';
import SpeakDisplay from './component/speakDisplay';
import CircularLoader from './component/CircularLoader';

function App() {
  const [microphoneAccess, setMicrophoneAccess] = useState(false);
  const [aiSpeaking, setAiSpeaking] = useState(false);
  const [record, setRecord] = useState(false);
  const [loading, setLoading] = useState(false);

  useLayoutEffect(() => {
    const checkAccess = async () => {
      const hasAccess = await checkUserAccesses(navigator);
      console.log('Microphone access:', hasAccess);
      setMicrophoneAccess(hasAccess);
    };

    checkAccess();
  }, []);
  
  const startRecording = () => {
    setRecord(true);
  }

  const stopRecording = () => {
    setRecord(false);
  }

  interface ReactMicStopEvent {
    blob: Blob;
    startTime: number;
    stopTime: number;
  }

  const onData = (recordedData: Blob) => {
    console.log('chunk of real-time data is: ', recordedData);
  }

  const onStop = async (recordedData: ReactMicStopEvent) => {
    setLoading(true);
    const audioUrl = URL.createObjectURL(recordedData.blob);
    const audio = new Audio(audioUrl);

    const response = await postUserSpeech(audio);
    setLoading(false);
    if (response) {
      setAiSpeaking(true);
      audio.play().then(() => {
        setAiSpeaking(false);
      });
    }
  }

  useEffect(() => {
    if (record) {
      // cut all audio playing
      document.querySelectorAll('audio').forEach(audio => audio.pause());
    }
  }, [record]);

  return (
    <section 
      className="w-full h-[100vh] flex flex-col items-center justify-start space-y-4"
      style={{ backgroundImage: 'linear-gradient(to left top, #845ec2, #8f68cd, #9971d8, #a47be3, #af85ee, #bb8ef3, #c697f7, #d1a0fc, #ddaafc, #e8b5fd, #f2c0fd, #faccff)' }}
    >
      <h1 className="text-4xl font-bold text-center text-white h-80 flex items-center justify-center">
        Welcome to TalkWithMe AI!
      </h1>
      <div
        className='w-[100px] h-[100px] bg-white rounded-full flex items-center justify-center transition-all ease-in-out duration-300'
        style={{
          backgroundColor: aiSpeaking ? 'white' : '#cec7c7',
          animation: aiSpeaking ? 'pulse 1s infinite' : 'none',
        }}
      >
        {loading ? <CircularLoader /> : <SpeakDisplay animate={aiSpeaking} />}
      </div>

      <ReactMic
        record={record}
        className='w-[350px] h-[100px] rounded-lg'
        visualSetting="sinewave"
        onStop={onStop}
        onData={onData}
        strokeColor="#ffffff"
        backgroundColor={aiSpeaking ? "#647185" : "#a45ead"}
        mimeType="audio/wav"
        // noiseSuppression={true}
      />

      <button
        className={
          'w-[250px] h-12 bg-white text-gray-700 font-bold rounded-lg hover:bg-gray-300 hover:w-[270px] hover:h-14 hover:rounded-2xl transition-all ease-in-out duration-300' +
          (record || !microphoneAccess ? 'opacity-50 cursor-not-allowed' : '')
        }
        onClick={() => (record ? stopRecording() : startRecording())}
        disabled={!microphoneAccess}
      >
        {
          record ? "I'm done" : "Let's talk!"
        }
      </button>

      {!microphoneAccess && (
        <p className="text-l text-red-500">
          You need to grant microphone permissions to use this feature
        </p>
      )}
    </section>
  );
}

export default App;
