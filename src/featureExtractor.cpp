//Name: Ahilesh Vadivel
//Date: 5th February 2026
// Feature Extractor for Content-Based Image Retrieval
//Description: Extracts visual features from all images in a directory and saves them to a CSV file for
//efficient querying.This is the offline feature extraction phase.

//Supported Feature Types:
//baseline:  7x7 center pixel region(147 features)
//histogram : 8x8x8 RGB color histogram(512 features)
//multi : Top / bottom spatial histograms(1024 features)
//texture : Color + gradient magnitude(528 features)
//extended (Task 4 extension): Color + magnitude + orientation + Laws filters(616 features)

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <filesystem>
#include <string>
#include "opencv2/opencv.hpp"
#include "csv_util.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

vector<float> extract7x7Features(Mat& image) {
    /*
      Extract 7x7 center pixel region as feature vector (Baseline method)
      Arguments:
        image - Input BGR color image (Mat)
      Returns:
        Vector of 147 normalized float values [B,G,R for each of 49 pixels]
    */
    vector<float> features;

    int centerX = image.cols / 2;
    int centerY = image.rows / 2;
    int startX = centerX - 3;
    int startY = centerY - 3;

    for (int y = startY; y < startY + 7; y++) {
        for (int x = startX; x < startX + 7; x++) {
            Vec3b pixel = image.at<Vec3b>(y, x);
            features.push_back(pixel[0] / 255.0f);  // Blue
            features.push_back(pixel[1] / 255.0f);  // Green
            features.push_back(pixel[2] / 255.0f);  // Red
        }
    }

    return features;
}

vector<float> extractRGBHistogram(Mat& image) {
    /*
      Extract whole-image RGB color histogram
      Arguments:
        image - Input BGR color image (Mat)
      Returns:
        Vector of 512 normalized float values (8x8x8 bins for R,G,B)
    */

    int bins = 8;  // 8 bins per channel
    int binSize = 256 / bins;  // 256/8 = 32 values per bin

    // Create 3D histogram (initialized to 0)
    vector<vector<vector<int>>> histogram(bins,
        vector<vector<int>>(bins,
            vector<int>(bins, 0)));

    // Count pixels into bins
    int totalPixels = 0;
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            Vec3b pixel = image.at<Vec3b>(y, x);

            // Determine which bin each channel falls into
            int b_bin = pixel[0] / binSize;  // Blue
            int g_bin = pixel[1] / binSize;  // Green
            int r_bin = pixel[2] / binSize;  // Red

            // Handle edge case: pixel value 255 goes to bin 7, not 8
            if (b_bin >= bins) b_bin = bins - 1;
            if (g_bin >= bins) g_bin = bins - 1;
            if (r_bin >= bins) r_bin = bins - 1;

            histogram[r_bin][g_bin][b_bin]++;
            totalPixels++;
        }
    }

    // Normalize and flatten to 1D vector
    vector<float> features;
    for (int r = 0; r < bins; r++) {
        for (int g = 0; g < bins; g++) {
            for (int b = 0; b < bins; b++) {
                // Normalize by total pixels
                float normalized = (float)histogram[r][g][b] / totalPixels;
                features.push_back(normalized);
            }
        }
    }

    return features;  // 512 elements (8x8x8)
}


vector<float> extractMultiHistogram(Mat& image) {
    /*
     Extract spatial color histograms from top and bottom image halves
     Arguments:
       image - Input BGR color image (Mat)
     Returns:
       Vector of 1024 normalized float values (512 for top + 512 for bottom)
    */

    int bins = 8;
    int binSize = 256 / bins;

    // Split image into top and bottom halves
    int midRow = image.rows / 2;
    Mat topHalf = image(Rect(0, 0, image.cols, midRow));
    Mat bottomHalf = image(Rect(0, midRow, image.cols, image.rows - midRow));

    vector<float> features;

    // Process both halves
    Mat halves[2] = { topHalf, bottomHalf };

    for (int h = 0; h < 2; h++) {
        Mat& half = halves[h];

        // Create 3D histogram for this half
        vector<vector<vector<int>>> histogram(bins,
            vector<vector<int>>(bins,
                vector<int>(bins, 0)));

        // Count pixels
        int totalPixels = 0;
        for (int y = 0; y < half.rows; y++) {
            for (int x = 0; x < half.cols; x++) {
                Vec3b pixel = half.at<Vec3b>(y, x);

                int b_bin = pixel[0] / binSize;
                int g_bin = pixel[1] / binSize;
                int r_bin = pixel[2] / binSize;

                if (b_bin >= bins) b_bin = bins - 1;
                if (g_bin >= bins) g_bin = bins - 1;
                if (r_bin >= bins) r_bin = bins - 1;

                histogram[r_bin][g_bin][b_bin]++;
                totalPixels++;
            }
        }

        // Normalize and add to features
        for (int r = 0; r < bins; r++) {
            for (int g = 0; g < bins; g++) {
                for (int b = 0; b < bins; b++) {
                    float normalized = (float)histogram[r][g][b] / totalPixels;
                    features.push_back(normalized);
                }
            }
        }
    }

    return features;  // 1024 elements (512 + 512)
}



int sobelX3x3(cv::Mat& src, cv::Mat& dst) {
    /*
    Compute Sobel X (horizontal) gradient using separable 3x3 kernel
    Arguments:
        src - Input BGR color image (CV_8UC3)
        dst - Output gradient image (CV_16SC3), passed by reference
    Returns:
        0 on success, -1 on error
    */

    dst.create(src.size(), CV_16SC3);
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    // Step 1: Apply [1,2,1] vertically
    for (int i = 1; i < src.rows - 1; i++) {
        short* tempRow = temp.ptr<short>(i);
        for (int j = 0; j < src.cols; j++) {
            cv::Vec3b top = src.ptr<cv::Vec3b>(i - 1)[j];
            cv::Vec3b mid = src.ptr<cv::Vec3b>(i)[j];
            cv::Vec3b bot = src.ptr<cv::Vec3b>(i + 1)[j];

            for (int c = 0; c < 3; c++) {
                tempRow[j * 3 + c] = top[c] + 2 * mid[c] + bot[c];
            }
        }
    }

    // Handle top and bottom borders
    for (int j = 0; j < src.cols; j++) {
        cv::Vec3b pixel_top = src.ptr<cv::Vec3b>(0)[j];
        cv::Vec3b pixel_bot = src.ptr<cv::Vec3b>(src.rows - 1)[j];
        short* temp_top = temp.ptr<short>(0);
        short* temp_bot = temp.ptr<short>(src.rows - 1);

        for (int c = 0; c < 3; c++) {
            temp_top[j * 3 + c] = pixel_top[c] * 4;
            temp_bot[j * 3 + c] = pixel_bot[c] * 4;
        }
    }

    // Step 2: Apply [-1,0,1] horizontally
    for (int i = 0; i < temp.rows; i++) {
        short* dstRow = dst.ptr<short>(i);
        short* tempRow = temp.ptr<short>(i);

        for (int j = 1; j < temp.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                short left = tempRow[(j - 1) * 3 + c];
                short right = tempRow[(j + 1) * 3 + c];
                dstRow[j * 3 + c] = -left + right;
            }
        }
    }

    // Handle left and right borders
    for (int i = 0; i < dst.rows; i++) {
        short* dstRow = dst.ptr<short>(i);
        for (int c = 0; c < 3; c++) {
            dstRow[0 * 3 + c] = 0;
            dstRow[(dst.cols - 1) * 3 + c] = 0;
        }
    }

    return 0;
}


int sobelY3x3(cv::Mat& src, cv::Mat& dst) {
    /*
      Compute Sobel Y (vertical) gradient using separable 3x3 kernel
      Arguments:
        src - Input BGR color image (CV_8UC3)
        dst - Output gradient image (CV_16SC3), passed by reference
      Returns:
        0 on success, -1 on error
    */

    dst.create(src.size(), CV_16SC3);
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    // Step 1: Apply [1,2,1] horizontally
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(i);
        short* tempRow = temp.ptr<short>(i);

        for (int j = 1; j < src.cols - 1; j++) {
            cv::Vec3b left = srcRow[j - 1];
            cv::Vec3b mid = srcRow[j];
            cv::Vec3b right = srcRow[j + 1];

            for (int c = 0; c < 3; c++) {
                tempRow[j * 3 + c] = left[c] + 2 * mid[c] + right[c];
            }
        }
    }

    // Handle left and right borders
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b pixel_left = src.ptr<cv::Vec3b>(i)[0];
        cv::Vec3b pixel_right = src.ptr<cv::Vec3b>(i)[src.cols - 1];
        short* tempRow = temp.ptr<short>(i);

        for (int c = 0; c < 3; c++) {
            tempRow[0 * 3 + c] = pixel_left[c] * 4;
            tempRow[(src.cols - 1) * 3 + c] = pixel_right[c] * 4;
        }
    }

    // Step 2: Apply [-1,0,1] vertically
    for (int i = 1; i < temp.rows - 1; i++) {
        short* dstRow = dst.ptr<short>(i);

        for (int j = 0; j < temp.cols; j++) {
            short* topRow = temp.ptr<short>(i - 1);
            short* botRow = temp.ptr<short>(i + 1);

            for (int c = 0; c < 3; c++) {
                short top = topRow[j * 3 + c];
                short bot = botRow[j * 3 + c];
                dstRow[j * 3 + c] = top - bot;
            }
        }
    }

    // Handle top and bottom borders
    for (int j = 0; j < dst.cols; j++) {
        short* top_row = dst.ptr<short>(0);
        short* bot_row = dst.ptr<short>(dst.rows - 1);

        for (int c = 0; c < 3; c++) {
            top_row[j * 3 + c] = 0;
            bot_row[j * 3 + c] = 0;
        }
    }

    return 0;
}


int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst) {
    /*
      Compute gradient magnitude from Sobel X and Y gradients
      Arguments:
        sx  - Sobel X gradient image (CV_16SC3)
        sy  - Sobel Y gradient image (CV_16SC3)
        dst - Output magnitude image (CV_8UC3), passed by reference
      Returns:
        0 on success, -1 on error
    */

    if (sx.size() != sy.size() || sx.type() != CV_16SC3 || sy.type() != CV_16SC3) {
        printf("Error: sx and sy must be same size and CV_16SC3\n");
        return -1;
    }

    dst.create(sx.size(), CV_8UC3);

    for (int i = 0; i < sx.rows; i++) {
        cv::Vec3s* sxRow = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s* syRow = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < sx.cols; j++) {
            for (int c = 0; c < 3; c++) {
                short gx = sxRow[j][c];
                short gy = syRow[j][c];

                double mag = sqrt(gx * gx + gy * gy);
                if (mag > 255) mag = 255;

                dstRow[j][c] = (uchar)mag;
            }
        }
    }

    return 0;
}


vector<float> extractTextureHistogram(Mat& image) {
    /*
      Extract gradient magnitude histogram for texture analysis
      Arguments:
        image - Input BGR color image (Mat)
      Returns:
        Vector of 16 normalized float values (histogram of gradient magnitudes)
    */

    int bins = 16;
    int binSize = 256 / bins;  // 16 values per bin

    // Compute Sobel gradients
    Mat sobelX, sobelY, mag;
    sobelX3x3(image, sobelX);
    sobelY3x3(image, sobelY);
    magnitude(sobelX, sobelY, mag);

    // Create histogram (1D for magnitude)
    vector<int> histogram(bins, 0);

    // Count magnitude values into bins
    int totalPixels = 0;
    for (int y = 0; y < mag.rows; y++) {
        for (int x = 0; x < mag.cols; x++) {
            Vec3b pixel = mag.at<Vec3b>(y, x);

            // Average the 3 channels for magnitude
            int avg_mag = (pixel[0] + pixel[1] + pixel[2]) / 3;

            int bin = avg_mag / binSize;
            if (bin >= bins) bin = bins - 1;

            histogram[bin]++;
            totalPixels++;
        }
    }

    // Normalize
    vector<float> features;
    for (int i = 0; i < bins; i++) {
        float normalized = (float)histogram[i] / totalPixels;
        features.push_back(normalized);
    }

    return features;  // 16 elements
}


vector<float> extractColorAndTexture(Mat& image) {
    /*
      Extract combined color and texture features
      Arguments:
        image - Input BGR color image (Mat)
      Returns:
        Vector of 528 normalized float values (512 color + 16 texture)
    */

    vector<float> features;

    // Get color histogram (512 bins)
    vector<float> color = extractRGBHistogram(image);

    // Get texture histogram (16 bins)
    vector<float> texture = extractTextureHistogram(image);

    // Combine them
    features.insert(features.end(), color.begin(), color.end());
    features.insert(features.end(), texture.begin(), texture.end());

    return features;  // 528 total
}


vector<float> extractOrientationHistogram(Mat& image) {
    /*
      Extract gradient orientation histogram
      Arguments:
        image - Input BGR color image (Mat)
      Returns:
        Vector of 8 normalized float values (8 orientation bins, 0-360 degrees)
    */

    int bins = 8;
    float binSize = 360.0f / bins;  // 45 degrees per bin

    // Compute Sobel gradients
    Mat sobelX, sobelY;
    sobelX3x3(image, sobelX);
    sobelY3x3(image, sobelY);

    // Create histogram
    vector<int> histogram(bins, 0);

    // Count orientations into bins
    int totalPixels = 0;
    for (int y = 0; y < sobelX.rows; y++) {
        for (int x = 0; x < sobelX.cols; x++) {
            Vec3s sx_pixel = sobelX.at<Vec3s>(y, x);
            Vec3s sy_pixel = sobelY.at<Vec3s>(y, x);

            // Average the 3 channels
            float gx = (sx_pixel[0] + sx_pixel[1] + sx_pixel[2]) / 3.0f;
            float gy = (sy_pixel[0] + sy_pixel[1] + sy_pixel[2]) / 3.0f;

            // Compute orientation in degrees (0-360)
            float angle = atan2(gy, gx) * 180.0f / CV_PI;  // Returns -180 to 180
            if (angle < 0) angle += 360.0f;  // Convert to 0-360

            // Determine bin
            int bin = (int)(angle / binSize);
            if (bin >= bins) bin = bins - 1;

            histogram[bin]++;
            totalPixels++;
        }
    }

    // Normalize
    vector<float> features;
    for (int i = 0; i < bins; i++) {
        float normalized = (float)histogram[i] / totalPixels;
        features.push_back(normalized);
    }

    return features;  // 8 elements
}


Mat createLawsFilter(vector<float>& kernel1, vector<float>& kernel2) {
    /*
      Create 2D Laws filter from two 1D kernels using outer product
      Arguments:
        kernel1 - First 1D kernel (typically 5 elements)
        kernel2 - Second 1D kernel (typically 5 elements)
      Returns:
        2D filter matrix (CV_32F)
    */

    int size = kernel1.size();
    Mat filter(size, size, CV_32F);

    // Outer product
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            filter.at<float>(i, j) = kernel1[i] * kernel2[j];
        }
    }

    return filter;
}


Mat applyLawsFilter(Mat& image, Mat& filter) {
    /*
      Apply a Laws filter to an image and return response
      Arguments:
        image  - Input BGR color image (Mat)
        filter - 2D Laws filter to apply (CV_32F)
      Returns:
        Filter response image (CV_32F), absolute values
    */

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Convert to float for filtering
    Mat imageFloat;
    gray.convertTo(imageFloat, CV_32F);

    // Apply filter using OpenCV's filter2D
    Mat response;
    filter2D(imageFloat, response, CV_32F, filter);

    // Take absolute value (we care about texture presence, not direction)
    response = abs(response);

    return response;
}


vector<float> extractLawsHistogram(Mat& response) {
    /*
      Extract histogram from a Laws filter response image
      Arguments:
        response - Laws filter response image (CV_32F)
      Returns:
        Vector of 16 normalized float values (histogram of response intensities)
    */

    int bins = 16;

    // Find min and max values for normalization
    double minVal, maxVal;
    minMaxLoc(response, &minVal, &maxVal);

    // Create histogram
    vector<int> histogram(bins, 0);

    int totalPixels = 0;
    for (int y = 0; y < response.rows; y++) {
        for (int x = 0; x < response.cols; x++) {
            float value = response.at<float>(y, x);

            // Normalize value to 0-1 range, then to bin index
            float normalized = (value - minVal) / (maxVal - minVal + 1e-6);
            int bin = (int)(normalized * bins);
            if (bin >= bins) bin = bins - 1;

            histogram[bin]++;
            totalPixels++;
        }
    }

    // Normalize histogram
    vector<float> features;
    for (int i = 0; i < bins; i++) {
        float norm = (float)histogram[i] / totalPixels;
        features.push_back(norm);
    }

    return features;  // 16 elements
}


vector<float> extractLawsFeatures(Mat& image) {
    /*
      Extract texture features using 5 different Laws filters
      Arguments:
        image - Input BGR color image (Mat)
      Returns:
        Vector of 80 normalized float values (5 filters × 16 bins each)
    */

    // Define the 5 basic 1D kernels
    vector<float> L5 = { 1, 4, 6, 4, 1 };           // Level
    vector<float> E5 = { -1, -2, 0, 2, 1 };         // Edge
    vector<float> S5 = { -1, 0, 2, 0, -1 };         // Spot

    // Create 2D filters
    Mat L5E5 = createLawsFilter(L5, E5);  // Horizontal edges
    Mat E5L5 = createLawsFilter(E5, L5);  // Vertical edges
    Mat E5E5 = createLawsFilter(E5, E5);  // Corners/spots
    Mat S5S5 = createLawsFilter(S5, S5);  // Grainy texture
    Mat E5S5 = createLawsFilter(E5, S5);  // Ripples

    vector<float> features;

    // Apply each filter and extract histogram
    Mat filters[5] = { L5E5, E5L5, E5E5, S5S5, E5S5 };

    for (int i = 0; i < 5; i++) {
        Mat response = applyLawsFilter(image, filters[i]);
        vector<float> hist = extractLawsHistogram(response);
        features.insert(features.end(), hist.begin(), hist.end());
    }

    return features;  // 80 elements (5 × 16)
}


vector<float> extractAllFeatures(Mat& image) {
    /*
      Extract comprehensive features: color, magnitude, orientation, and Laws
      Arguments:
        image - Input BGR color image (Mat)
      Returns:
        Vector of 616 normalized float values (512+16+8+80)
    */

    vector<float> features;

    // Color histogram (512)
    vector<float> color = extractRGBHistogram(image);
    features.insert(features.end(), color.begin(), color.end());

    // Gradient magnitude histogram (16)
    vector<float> magnitude = extractTextureHistogram(image);
    features.insert(features.end(), magnitude.begin(), magnitude.end());

    // Gradient orientation histogram (8)
    vector<float> orientation = extractOrientationHistogram(image);
    features.insert(features.end(), orientation.begin(), orientation.end());

    // Laws filter features (80)
    vector<float> laws = extractLawsFeatures(image);
    features.insert(features.end(), laws.begin(), laws.end());

    return features;  // 616 total
}


int main(int argc, char* argv[]) {
    char csv_filename[256];
    int image_count = 0;
    string feature_type = "baseline";  // Default

    // Check command line arguments
    if (argc < 3) {
        printf("Usage: %s <image_directory> <output_csv_file> [feature_type]\n", argv[0]);
        printf("  feature_type: 'baseline' or 'histogram' (default: baseline)\n");
        return -1;
    }

    string dirname = argv[1];
    strcpy(csv_filename, argv[2]);

    if (argc >= 4) {
        feature_type = argv[3];
    }

    printf("Processing directory: %s\n", dirname.c_str());
    printf("Output CSV file: %s\n", csv_filename);
    printf("Feature type: %s\n\n", feature_type.c_str());

    // Check if directory exists
    if (!fs::exists(dirname) || !fs::is_directory(dirname)) {
        printf("Cannot open directory %s\n", dirname.c_str());
        return -1;
    }

    // Loop over all files in directory
    for (const auto& entry : fs::directory_iterator(dirname)) {

        if (!entry.is_regular_file()) {
            continue;
        }

        string filepath = entry.path().string();
        string filename = entry.path().filename().string();
        string extension = entry.path().extension().string();

        // Check if file is an image
        if (extension == ".jpg" || extension == ".jpeg" ||
            extension == ".png" || extension == ".ppm" ||
            extension == ".tif" || extension == ".tiff") {

            printf("Processing: %s\n", filename.c_str());

            // Read the image
            Mat image = imread(filepath);
            if (image.empty()) {
                printf("Unable to read image %s\n", filepath.c_str());
                continue;
            }

            // Extract features based on type
            vector<float> features;
            if (feature_type == "histogram") {
                features = extractRGBHistogram(image);
            }
            else if (feature_type == "multi") {
                features = extractMultiHistogram(image);
            }
            else if (feature_type == "texture") {
                features = extractColorAndTexture(image);
            }
            else if (feature_type == "extended") { 
                features = extractAllFeatures(image);
            }
            else {
                features = extract7x7Features(image);
            }

            printf("  Extracted %lu features\n", features.size());

            // Convert filename to char* for csv function
            char* fname_cstr = new char[filename.length() + 1];
            strcpy(fname_cstr, filename.c_str());

            // Write to CSV
            int reset = (image_count == 0) ? 1 : 0;
            append_image_data_csv(csv_filename, fname_cstr, features, reset);

            delete[] fname_cstr;
            image_count++;
        }
    }

    printf("\nProcessed %d images\n", image_count);
    printf("Features saved to %s\n", csv_filename);

    return 0;
}