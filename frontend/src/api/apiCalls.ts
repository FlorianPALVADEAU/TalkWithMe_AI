import axios from "axios";

const axiosClient = axios.create({
    baseURL: "http://localhost:8000/"
});

export const postUserSpeech = async (recordedBlob: Blob): Promise<Blob | undefined> => {
    console.log("postUserSpeech called with:", recordedBlob);
    console.log("Type:", typeof recordedBlob);
    console.log("Is Blob?", recordedBlob instanceof Blob);
    
    if (!recordedBlob || !(recordedBlob instanceof Blob)) {
        console.error("Invalid blob received:", recordedBlob);
        throw new Error("audioBlob is not a Blob or File");
    }

    try {
        const formData = new FormData();
        formData.append("audio", recordedBlob, "recording.webm");
        
        console.log("FormData created with audio blob");
        
        const res = await axiosClient.post("/api/process-audio/", formData, {
            headers: {
                "Content-Type": "multipart/form-data",
            },
            responseType: 'blob'
        });

        console.log("Response received:", res);
        return res.data;
    } catch (error) {
        console.error("Error while processing audio:", error);
        alert("Error while processing audio");
        return undefined;
    }
};