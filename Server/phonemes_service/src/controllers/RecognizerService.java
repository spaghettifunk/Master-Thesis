package controllers;

import edu.cmu.sphinx.api.Configuration;
import edu.cmu.sphinx.api.StreamSpeechRecognizer;

import java.io.IOException;

public class RecognizerService {

    // CMU variables
    public static Configuration configuration = null;
    public static StreamSpeechRecognizer recognizer = null;

    public static void InitializeRecognitionSystem() throws IOException {

        System.out.println("Loading models...");

        configuration = new Configuration();

        // Load model from the jar
        configuration.setAcousticModelPath("model/en-us");
        configuration.setDictionaryPath("model/cmudict-en-us.dict");
        configuration.setLanguageModelPath("model/en-us.lm.bin");

        recognizer = new StreamSpeechRecognizer(configuration);
    }
}
