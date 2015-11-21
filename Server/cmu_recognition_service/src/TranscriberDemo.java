/*
 * Copyright 1999-2013 Carnegie Mellon University.
 * Portions Copyright 2004 Sun Microsystems, Inc.
 * Portions Copyright 2004 Mitsubishi Electric Research Laboratories.
 * All Rights Reserved.  Use is subject to license terms.
 *
 * See the file "license.terms" for information on usage and
 * redistribution of this file, and for a DISCLAIMER OF ALL
 * WARRANTIES.
 */

import java.io.InputStream;

import edu.cmu.sphinx.api.Configuration;
import edu.cmu.sphinx.api.Context;
import edu.cmu.sphinx.api.SpeechResult;
import edu.cmu.sphinx.api.StreamSpeechRecognizer;
import edu.cmu.sphinx.decoder.adaptation.Stats;
import edu.cmu.sphinx.decoder.adaptation.Transform;
import edu.cmu.sphinx.recognizer.Recognizer;
import edu.cmu.sphinx.result.Result;
import edu.cmu.sphinx.result.WordResult;
import edu.cmu.sphinx.util.TimeFrame;

/**
 * A simple example that shows how to transcribe a continuous audio file that
 * has multiple utterances in it.
 */
public class TranscriberDemo {

    public static void main(String[] args) throws Exception {
        System.out.println("Loading models...");

        Configuration configuration = new Configuration();

        // Load model from the jar
        configuration.setAcousticModelPath("model/en-us");

        // You can also load model from folder
        // configuration.setAcousticModelPath("file:en-us");

        configuration.setDictionaryPath("model/cmudict-en-us.dict");
        configuration.setLanguageModelPath("model/en-us.lm.bin");

        StreamSpeechRecognizer recognizer = new StreamSpeechRecognizer(configuration);
        InputStream stream = TranscriberDemo.class.getResourceAsStream("__test.wav");
        stream.skip(44);

        // Simple recognition with generic model
        recognizer.startRecognition(stream);
        SpeechResult result;
        while ((result = recognizer.getResult()) != null) {

            System.out.format("Hypothesis: %s\n", result.getHypothesis());

            System.out.println("List of recognized words and their times:");
            for (WordResult r : result.getWords()) {
                System.out.println(r);
            }

            System.out.println("Best 3 hypothesis:");
            for (String s : result.getNbest(3))
                System.out.println(s);

        }
        recognizer.stopRecognition();

        // Live adaptation to speaker with speaker profiles

        stream = TranscriberDemo.class.getResourceAsStream("__test.wav");
        stream.skip(44);

        // Stats class is used to collect speaker-specific data
        Stats stats = recognizer.createStats(1);
        recognizer.startRecognition(stream);
        while ((result = recognizer.getResult()) != null) {
            stats.collect(result);
        }
        recognizer.stopRecognition();

        // Transform represents the speech profile
        Transform transform = stats.createTransform();
        recognizer.setTransform(transform);

        // Decode again with updated transform
        stream = TranscriberDemo.class.getResourceAsStream("__test.wav");
        stream.skip(44);
        recognizer.startRecognition(stream);
        while ((result = recognizer.getResult()) != null) {
            System.out.format("Hypothesis: %s\n", result.getHypothesis());
        }
        recognizer.stopRecognition();

        Context context = new Context(configuration);
        context.setLocalProperty("decoder->searchManager", "allphoneSearchManager");
        stream = TranscriberDemo.class.getResourceAsStream("__test.wav");
        stream.skip(44);

        // Simple recognition with generic model
        Recognizer phone_recognizer = context.getInstance(Recognizer.class);
        phone_recognizer.allocate();
        context.setSpeechSource(stream, TimeFrame.INFINITE);
        Result phone_result;
        while ((phone_result = phone_recognizer.recognize()) != null) {
            SpeechResult speechResult = new SpeechResult(phone_result);
            System.out.format("Hypothesis: %s\n", speechResult.getHypothesis());

            System.out.println("List of recognized words and their times:");
            for (WordResult r : speechResult.getWords()) {
                System.out.println(r);
            }

            System.out.println("Lattice contains " + speechResult.getLattice().getNodes().size() + " nodes");
        }
        phone_recognizer.deallocate();
    }
}
