package controllers;

import edu.cmu.sphinx.api.Configuration;
import edu.cmu.sphinx.api.StreamSpeechRecognizer;

import javax.servlet.ServletContext;
import javax.servlet.ServletContextEvent;
import javax.servlet.ServletContextListener;
import javax.servlet.annotation.WebListener;
import javax.servlet.http.HttpSessionAttributeListener;
import javax.servlet.http.HttpSessionEvent;
import javax.servlet.http.HttpSessionListener;
import javax.servlet.http.HttpSessionBindingEvent;
import java.io.IOException;
import java.net.URL;

@WebListener()
public class Listener implements ServletContextListener, HttpSessionListener, HttpSessionAttributeListener {

    // Public constructor is required by servlet spec
    public Listener() {
    }

    // -------------------------------------------------------
    // ServletContextListener implementation
    // -------------------------------------------------------
    public void contextInitialized(ServletContextEvent sce) {
      /* This method is called when the servlet context is
         initialized(when the Web application is deployed). 
         You can initialize servlet context related data here.
      */

        // CMU variables
        Configuration configuration = null;
        StreamSpeechRecognizer recognizer = null;

        System.out.println("Loading models...");

        try {
            configuration = new Configuration();

            // Load model from the jar
            URL modelPath = getClass().getClassLoader().getResource("model/en-us");
            URL dictionaryPath = getClass().getClassLoader().getResource("model/cmudict-en-us.dict");
            URL languageModelPath = getClass().getClassLoader().getResource("model/en-us.lm.bin");

            configuration.setAcousticModelPath(modelPath.getPath());
            configuration.setDictionaryPath(dictionaryPath.getPath());
            configuration.setLanguageModelPath(languageModelPath.getPath());

            recognizer = new StreamSpeechRecognizer(configuration);

            ServletContext context = sce.getServletContext();
            context.setAttribute("configuration", configuration);
            context.setAttribute("recognizer", recognizer);

        } catch (IOException e) {

        }
    }

    public void contextDestroyed(ServletContextEvent sce) {
      /* This method is invoked when the Servlet Context 
         (the Web application) is undeployed or 
         Application Server shuts down.
      */

        ServletContext context = sce.getServletContext();
        context.removeAttribute("configuration");
        context.removeAttribute("recognizer");
    }

    // -------------------------------------------------------
    // HttpSessionListener implementation
    // -------------------------------------------------------
    public void sessionCreated(HttpSessionEvent se) {
      /* Session is created. */
    }

    public void sessionDestroyed(HttpSessionEvent se) {
      /* Session is destroyed. */
    }

    // -------------------------------------------------------
    // HttpSessionAttributeListener implementation
    // -------------------------------------------------------

    public void attributeAdded(HttpSessionBindingEvent sbe) {
      /* This method is called when an attribute 
         is added to a session.
      */
    }

    public void attributeRemoved(HttpSessionBindingEvent sbe) {
      /* This method is called when an attribute
         is removed from a session.
      */
    }

    public void attributeReplaced(HttpSessionBindingEvent sbe) {
      /* This method is invoked when an attribute
         is replaced in a session.
      */
    }
}
