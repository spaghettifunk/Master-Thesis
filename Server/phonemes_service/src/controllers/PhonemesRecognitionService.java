package controllers;

import com.sun.net.httpserver.HttpServer;

import java.net.URI;

import org.glassfish.jersey.jdkhttp.JdkHttpServerFactory;
import org.glassfish.jersey.server.ResourceConfig;

import java.io.IOException;
import java.net.URISyntaxException;


// The Java class will be hosted at the URI path "/phonemes"
public class PhonemesRecognitionService {

    private static final String BASE_URI = "http://127.0.0.1:9099/";

    public static void main(String[] args) throws IOException, URISyntaxException {

        URI endpoint = new URI(BASE_URI);

        ResourceConfig transcriptionConfig = new ResourceConfig(Transcription.class);
        HttpServer server = JdkHttpServerFactory.createHttpServer(endpoint, transcriptionConfig);

        // initialize Sphinx
        RecognizerService.InitializeRecognitionSystem();

        System.out.println("Server running");
        System.out.println("Visit: http://127.0.0.1:9099/phonemes");
        System.out.println("Hit return to stop...");
        System.in.read();
        System.out.println("Stopping server");
        server.stop(0);
        System.out.println("Server stopped");
    }
}
