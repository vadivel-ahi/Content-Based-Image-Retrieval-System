/*
  Bruce A. Maxwell
  S21

  Sample code to identify image files in a directory
  Windows-compatible version using C++17 filesystem
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

/*
  Given a directory on the command line, scans through the directory for image files.
  Prints out the full path name for each file. This can be used as an argument to fopen or to cv::imread.
*/
int main(int argc, char* argv[]) {

    // Check for sufficient arguments
    if (argc < 2) {
        printf("usage: %s <directory path>\n", argv[0]);
        return -1;
    }

    // Get the directory path
    std::string dirname = argv[1];
    printf("Processing directory %s\n", dirname.c_str());

    // Check if directory exists
    if (!fs::exists(dirname) || !fs::is_directory(dirname)) {
        printf("Cannot open directory %s\n", dirname.c_str());
        return -1;
    }

    // Loop over all the files in the directory
    for (const auto& entry : fs::directory_iterator(dirname)) {

        // Check if it's a regular file
        if (!entry.is_regular_file()) {
            continue;
        }

        // Get filename and extension
        std::string filename = entry.path().filename().string();
        std::string extension = entry.path().extension().string();
        std::string fullpath = entry.path().string();

        // Check if the file is an image
        if (extension == ".jpg" || extension == ".jpeg" ||
            extension == ".png" ||
            extension == ".ppm" ||
            extension == ".tif" || extension == ".tiff") {

            printf("processing image file: %s\n", filename.c_str());
            printf("full path name: %s\n", fullpath.c_str());
        }
    }

    printf("Terminating\n");

    return 0;
}