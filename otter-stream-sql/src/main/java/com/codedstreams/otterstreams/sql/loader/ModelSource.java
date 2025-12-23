package com.codedstreams.otterstreams.sql.loader;

import java.io.InputStream;

/**
 * Interface for different model source implementations.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public interface ModelSource {

    /**
     * Opens an input stream to read the model.
     *
     * @return InputStream to model data
     * @throws Exception if model cannot be loaded
     */
    InputStream openStream() throws Exception;

    /**
     * Gets the source location/URI.
     *
     * @return source location string
     */
    String getLocation();

    /**
     * Checks if the model source is accessible.
     *
     * @return true if accessible, false otherwise
     */
    boolean isAccessible();

    /**
     * Gets the size of the model in bytes if available.
     *
     * @return size in bytes, or -1 if unknown
     */
    long getSize();

    /**
     * Closes any resources associated with this source.
     */
    void close();
}
