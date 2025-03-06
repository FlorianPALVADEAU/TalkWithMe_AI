import { useEffect, useLayoutEffect, useState } from 'react';
import './styles/App.css';
import checkUserAccesses from './helpers/checkUserAccesses';
import { ReactMic } from 'react-mic';
import { postUserSpeech } from './api/apiCalls';

import CircularLoader from './component/CircularLoader';
import SpeakDisplay from './component/SpeakDisplay';


function App() {
  const [microphoneAccess, setMicrophoneAccess] = useState(false);
  const [aiSpeaking, setAiSpeaking] = useState(false);
  const [record, setRecord] = useState(false);
  const [loading, setLoading] = useState(false);

  useLayoutEffect(() => {
    const checkAccess = async () => {
      const hasAccess = await checkUserAccesses(navigator);
      setMicrophoneAccess(hasAccess);
    };

    checkAccess();
  }, []);

  const startRecording = () => {
    setRecord(true);
  };

  const stopRecording = () => {
    setRecord(false);
  };

  interface ReactMicStopEvent {
    blob: Blob;
    startTime: number;
    stopTime: number;
  }

  const onStop = async (recordedData: ReactMicStopEvent) => {
    console.log('onStop called with:', recordedData);
    console.log('Blob type:', recordedData.blob.type);
    console.log('Is Blob?', recordedData.blob instanceof Blob);

    setLoading(true);


    try {
      const responseBlob = await postUserSpeech(recordedData.blob);

      if (responseBlob) {
        setAiSpeaking(true);
        const audioUrl = URL.createObjectURL(responseBlob);
        const audio = new Audio(audioUrl);
        audio.onended = () => {
          setAiSpeaking(false);
        };
        audio.play().catch((err) => {
          console.error('Error playing audio:', err);
          setAiSpeaking(false);
        });
      }
    } catch (error) {
      console.error('Error in onStop:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (record) {
      document.querySelectorAll('audio').forEach((audio) => audio.pause());
    }
  }, [record]);

  return (
    <section
      className='w-full h-[100vh] flex flex-col items-center justify-start space-y-4'
      style={{
        backgroundImage:
          'linear-gradient(to left top, #845ec2, #8f68cd, #9971d8, #a47be3, #af85ee, #bb8ef3, #c697f7, #d1a0fc, #ddaafc, #e8b5fd, #f2c0fd, #faccff)',
      }}
    >
      <h1 className='text-4xl font-bold text-center text-white h-80 flex items-center justify-center'>
        Bienvenue, je suis Louis le FouFou, poses moi une question sinon conséquences...
      </h1>
      <div
        className='w-[100px] h-[100px] bg-white rounded-full flex items-center justify-center transition-all ease-in-out duration-300'
        style={{
          backgroundColor: aiSpeaking ? 'white' : '#cec7c7',
          animation: aiSpeaking ? 'pulse 1s infinite' : loading ? 'distortCircle 1s infinite ease-in-out' : 'none',
          clipPath: 'circle(50% at 50% 50%)',
        }}
      >
        {loading ? <CircularLoader /> : <SpeakDisplay animate={aiSpeaking} />}
      </div>

      <ReactMic
        record={record}
        className='w-[350px] h-[100px] rounded-lg'
        visualSetting='frequencyBars'
        onStop={onStop}
        strokeColor='#ffffff'
        backgroundColor={aiSpeaking ? '#647185' : '#a45ead'}
        mimeType='audio/wav'
      />

      <button
        className={
          'w-[250px] h-12 bg-white text-gray-700 font-bold rounded-lg hover:bg-gray-300 hover:w-[270px] hover:h-14 hover:rounded-2xl transition-all ease-in-out duration-300' +
          (!microphoneAccess ? 'opacity-50 cursor-not-allowed' : '')
        }
        onClick={() => (record ? stopRecording() : startRecording())}
        disabled={!microphoneAccess}
      >
        {record ? "I'm done" : "Let's talk!"}
      </button>

      {!microphoneAccess && (
        <p className='text-l text-red-500'>
          You need to grant microphone permissions to use this feature
        </p>
      )}
    </section>
  );
}

export default App;
