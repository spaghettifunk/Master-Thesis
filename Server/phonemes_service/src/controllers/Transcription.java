package controllers;

import org.apache.commons.codec.binary.Base64;
import org.json.*;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import java.io.*;
import java.util.UUID;

import edu.cmu.sphinx.recognizer.Recognizer;
import edu.cmu.sphinx.api.Context;
import edu.cmu.sphinx.api.SpeechResult;
import edu.cmu.sphinx.decoder.adaptation.Stats;
import edu.cmu.sphinx.decoder.adaptation.Transform;
import edu.cmu.sphinx.result.Result;
import edu.cmu.sphinx.util.TimeFrame;


@Path("/phonemes")
public class Transcription {

    @POST
    @Path("/transcription")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public String getTranscription(String body) throws IOException {

        try {
            JSONObject requestObject = new JSONObject(body);
            String username = requestObject.getString("User");
            String audioFileString = requestObject.getString("FileAudio");

            String filename = "recorded_" + UUID.randomUUID() + ".wav";
            String editedFilename = "edited_" + filename;
            byte[] decoded = Base64.decodeBase64(audioFileString);

            // recorded file from the user
            File file = new File(filename);
            FileOutputStream os = new FileOutputStream(file);
            os.write(decoded);
            os.close();

            // TODO: check this part when deploying to VM
            Runtime runtime = Runtime.getRuntime();
            String command = "sox -r 8000 -c 1 " + filename + " " + editedFilename;
            Process soxprocess = runtime.exec(command);
            soxprocess.waitFor();

            // edited file with sox
            File editedFile = new File(editedFilename);

            String phonemes = extractPhonemes(editedFile.getAbsolutePath());
            file.delete();
            editedFile.delete();

            return "{ \"Response\" : \"SUCCESS\", \"Phonemes\" : \"" + phonemes.replace("SIL", "") + "\" }";
        } catch (Exception e) {
            return "{ \"Response\" : \"FAILED\", \"Reason\" : " + e.getMessage() + "\" }";
        }
    }

    private String extractPhonemes(String audioFile) throws Exception {

        InputStream stream = new FileInputStream(audioFile);
        stream.skip(44);

        // Simple recognition with generic model
        SpeechResult result;
        RecognizerService.recognizer.startRecognition(stream);

        // Live adaptation to speaker with speaker profiles
        // Stats class is used to collect speaker-specific data
        Stats stats = RecognizerService.recognizer.createStats(1);
        while ((result = RecognizerService.recognizer.getResult()) != null) {
            stats.collect(result);
        }

        // Transform represents the speech profile
        Transform transform = stats.createTransform();
        RecognizerService.recognizer.setTransform(transform);

        Context context = new Context(RecognizerService.configuration);
        context.setLocalProperty("decoder->searchManager", "allphoneSearchManager");

        // Simple recognition with generic model
        stream = new FileInputStream(audioFile);
        stream.skip(44);

        Recognizer phone_recognizer = context.getInstance(Recognizer.class);
        phone_recognizer.allocate();
        context.setSpeechSource(stream, TimeFrame.INFINITE);
        Result phone_result;
        String phonemes = "";
        while ((phone_result = phone_recognizer.recognize()) != null) {
            SpeechResult speechResult = new SpeechResult(phone_result);
            phonemes = speechResult.getHypothesis();
        }

        phone_recognizer.deallocate();
        RecognizerService.recognizer.stopRecognition();

        return phonemes;
    }
}
