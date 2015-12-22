package controllers;

import edu.cmu.sphinx.api.Configuration;
import edu.cmu.sphinx.api.Context;
import edu.cmu.sphinx.api.SpeechResult;
import edu.cmu.sphinx.api.StreamSpeechRecognizer;
import edu.cmu.sphinx.decoder.adaptation.Stats;
import edu.cmu.sphinx.decoder.adaptation.Transform;
import edu.cmu.sphinx.recognizer.Recognizer;
import edu.cmu.sphinx.result.Result;
import edu.cmu.sphinx.util.TimeFrame;

import javax.servlet.ServletContext;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.commons.codec.binary.Base64;
import org.json.*;

import java.io.*;
import java.util.UUID;


@WebServlet(name = "PhonemesService", urlPatterns = "/phonemes/transcription")
public class ServiceServlet extends HttpServlet {

    @javax.ws.rs.core.Context
    private ServletContext context;

    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        resp.getWriter().write("Ciao a tutti belli e brutti");
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {

        StringBuffer jb = new StringBuffer();
        String line = null;

        try {
            BufferedReader reader = request.getReader();
            while ((line = reader.readLine()) != null) {
                jb.append(line);
            }

            // get context
            if (this.context == null)
                this.context = getServletContext();

            String result = getTranscription(jb.toString());
            response.getWriter().write(result);

        } catch (Exception e) {
            /*report an error*/
        }
    }

    public String getTranscription(String body) throws IOException {

        OUTER:
        try {
            JSONObject requestObject = new JSONObject(body);
            String username = requestObject.getString("User");
            String audioFileString = requestObject.getString("FileAudio");

            String filename = "recorded_" + UUID.randomUUID() + ".wav";
            String editedFilename = "edited_" + filename;
            byte[] decoded = Base64.decodeBase64(audioFileString);

            // recorded file from the user
            File file = new File(context.getRealPath(filename));
            FileOutputStream os = new FileOutputStream(file);
            os.write(decoded);
            os.close();

            // TODO: check this part when deploying to VM
            Runtime runtime = Runtime.getRuntime();
            String command = "/usr/bin/sox -c 1 " + context.getRealPath(filename) + " " + context.getRealPath(editedFilename);
            Process soxprocess = runtime.exec(command);
            soxprocess.waitFor();

            // edited file with sox
            File editedFile = new File(context.getRealPath(editedFilename));

            String phonemes = extractPhonemes(editedFile.getAbsolutePath());
            phonemes = phonemes.replace("+SPN+", "");
            phonemes = phonemes.replace("SIL", "");

            System.out.println("PHONEMES are: " + phonemes);

            return "{ \"Response\" : \"SUCCESS\", \"Phonemes\" : \"" + phonemes + "\" }";
        } catch (Exception e) {
            break OUTER;
        }
        return "{ \"Response\" : \"FAILED\", \"Reason\" : Cazzo ne so \" }";
    }

    private String extractPhonemes(String audioFile) throws Exception {

        InputStream stream = new FileInputStream(audioFile);
        stream.skip(44);

        System.out.println("**** START PHONEME RECOGNITION ****");

        // Simple recognition with generic model
        SpeechResult result;
        StreamSpeechRecognizer recognizer = (StreamSpeechRecognizer) context.getAttribute("recognizer");
        Configuration configuration = (Configuration) context.getAttribute("configuration");

        System.out.println("**** SET UP MODEL ****");

        recognizer.startRecognition(stream);

        // Live adaptation to speaker with speaker profiles
        // Stats class is used to collect speaker-specific data
        Stats stats = recognizer.createStats(1);
        while ((result = recognizer.getResult()) != null) {
            stats.collect(result);
        }

        System.out.println("**** MODEL ADAPTED ****");

        // Transform represents the speech profile
        Transform transform = stats.createTransform();
        recognizer.setTransform(transform);

        Context context = new Context(configuration);
        context.setLocalProperty("decoder->searchManager", "allphoneSearchManager");

        System.out.println("**** SET DECODER ****");

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

        System.out.println("**** FINISHED ****");

        phone_recognizer.deallocate();
        recognizer.stopRecognition();

        System.out.println("**** STOP PHONEME RECOGNITION ****");

        return phonemes;
    }

}
