package com.codedstreams.otterstreams.sql.loader;

import java.io.InputStream;

public interface ModelLoader {
    InputStream loadModel() throws Exception;
    String getModelPath();
}
