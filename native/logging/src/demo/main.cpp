#include <iostream>

#include "logging/logging.h"

#define LOG_TAG "ActinisLoggingDemo"

int main() {
    LOG_DEBUG(LOG_TAG, "Debug log %d", 1);
    LOG_INFO(LOG_TAG, "Info log %f", 1.0f);
    LOG_WARN(LOG_TAG, "Warning log \"%s\"", "string arg");
    LOG_ERROR(LOG_TAG, "Error log \"%s\"", "string arg");

    return 0;
}
