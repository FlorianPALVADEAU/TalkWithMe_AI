import axios from "axios";

const axiosClient = axios.create({
    baseURL: "http://localhost:8000"
});



export const postUserSpeech = async (audio: HTMLAudioElement): Promise<HTMLAudioElement | undefined> => {
    if (!audio) return undefined;
    try {
        // send audio file
        const res = await axiosClient.post("/api/process-audio", audio, {
            headers: {
                "Content-Type": "audio/wav",
            },
        });
        return res.data;
    } catch (error) {
        alert("Error while processing audio");
        console.error("Error while processing audio:", error);
        return undefined;
    }
};