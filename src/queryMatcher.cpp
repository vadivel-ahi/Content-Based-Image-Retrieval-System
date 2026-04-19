//Name: Ahilesh Vadivel
//Date: 5th February 2026
// Query Matcher for Content-Based Image Retrieval
//Description: Finds the N most similar images to a target image by computing distances between
//feature vectors.This is the online query / matching phase.

//Supported Distance Metrics :
//ssd : Sum of squared differences(for baseline)
//histogram : Histogram intersection(for color histograms)
//multi : Multi - region histogram intersection(for spatial histograms)
//texture : Combined color + texture matching
//extended : Multi - feature matching(color + magnitude + orientation + Laws)
//embedding : Cosine distance(for ResNet18 deep features)
//car : Custom hybrid matching(DNN + circles + aspect + texture + horizontal lines)

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>
#include "opencv2/opencv.hpp"
#include "csv_util.h"

using namespace cv;
using namespace std;


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
            features.push_back(pixel[0] / 255.0f);
            features.push_back(pixel[1] / 255.0f);
            features.push_back(pixel[2] / 255.0f);
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

    int bins = 8;
    int binSize = 256 / bins;

    vector<vector<vector<int>>> histogram(bins,
        vector<vector<int>>(bins,
            vector<int>(bins, 0)));

    int totalPixels = 0;
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            Vec3b pixel = image.at<Vec3b>(y, x);

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

    vector<float> features;
    for (int r = 0; r < bins; r++) {
        for (int g = 0; g < bins; g++) {
            for (int b = 0; b < bins; b++) {
                float normalized = (float)histogram[r][g][b] / totalPixels;
                features.push_back(normalized);
            }
        }
    }

    return features;
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


// =====================================================
// CUSTOM CAR DETECTION FEATURES
// =====================================================
float detectCircles(Mat& image) {
    /*
      Detect circular structures in image using Hough Circle Transform
      Arguments:
        image - Input BGR color image (Mat)
      Returns:
        Normalized circle score (0.0 to 1.0, based on detected circle count / 4)
    */

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Apply Gaussian blur to reduce noise
    GaussianBlur(gray, gray, Size(9, 9), 2, 2);

    vector<Vec3f> circles;

    // Hough Circle Transform
    // Parameters tuned for car wheels
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
        gray.rows / 8,  // min distance between circles
        100,            // Canny edge threshold
        30,             // accumulator threshold
        20,             // min radius (pixels)
        100);           // max radius (pixels)

    // Normalize circle count (assume max 4 wheels visible)
    float circle_score = min((float)circles.size() / 4.0f, 1.0f);

    printf("  Detected %lu circles (score: %.3f)\n", circles.size(), circle_score);

    return circle_score;
}

float getAspectRatio(Mat& image) {
    /*
      Compute image aspect ratio (width/height)

      Arguments:
        image - Input BGR color image (Mat)

      Returns:
        Aspect ratio as float (>1.0 for landscape, <1.0 for portrait)
    */

    float ratio = (float)image.cols / (float)image.rows;
    printf("  Aspect ratio: %.3f\n", ratio);
    return ratio;
}


vector<float> extractLowerRegionTexture(Mat& image) {
    /*
      Extract gradient magnitude histogram from lower 40% of image

      Arguments:
        image - Input BGR color image (Mat)

      Returns:
        Vector of 16 normalized float values (texture in lower region)
    */

    int startRow = (int)(image.rows * 0.6);  // Start at 60% down
    Mat lowerRegion = image(Rect(0, startRow, image.cols, image.rows - startRow));

    // Compute gradient magnitude on lower region
    Mat sobelX, sobelY, mag;
    sobelX3x3(lowerRegion, sobelX);
    sobelY3x3(lowerRegion, sobelY);
    magnitude(sobelX, sobelY, mag);

    // Create histogram
    int bins = 16;
    int binSize = 256 / bins;
    vector<int> histogram(bins, 0);

    int totalPixels = 0;
    for (int y = 0; y < mag.rows; y++) {
        for (int x = 0; x < mag.cols; x++) {
            Vec3b pixel = mag.at<Vec3b>(y, x);
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
        features.push_back((float)histogram[i] / totalPixels);
    }

    return features;
}


vector<float> extractMetallicTexture(Mat& image) {
    /*
      Extract texture features from middle 40% of image using Laws filters

      Arguments:
        image - Input BGR color image (Mat)

      Returns:
        Vector of 32 normalized float values (2 Laws filters × 16 bins)
    */

    // Extract middle 40% of image (where car body typically is)
    int startRow = (int)(image.rows * 0.3);
    int endRow = (int)(image.rows * 0.7);
    Mat midRegion = image(Rect(0, startRow, image.cols, endRow - startRow));

    // Use Laws L5L5 (smooth) and E5E5 (edges) filters
    vector<float> L5 = { 1, 4, 6, 4, 1 };
    vector<float> E5 = { -1, -2, 0, 2, 1 };

    Mat L5L5 = createLawsFilter(L5, L5);  // Smooth regions
    Mat E5E5 = createLawsFilter(E5, E5);  // Edge patterns

    vector<float> features;
    Mat filters[2] = { L5L5, E5E5 };

    for (int i = 0; i < 2; i++) {
        Mat response = applyLawsFilter(midRegion, filters[i]);
        vector<float> hist = extractLawsHistogram(response);
        features.insert(features.end(), hist.begin(), hist.end());
    }

    return features;  // 32 features (2 filters × 16 bins)
}


float getHorizontalLineDensity(Mat& image) {
    /*
      Measure density of horizontal edges in image (car profile characteristic)

      Arguments:
        image - Input BGR color image (Mat)

      Returns:
        Normalized horizontal edge density (0.0 to 1.0)
    */

    Mat sobelX, sobelY;
    sobelX3x3(image, sobelX);
    sobelY3x3(image, sobelY);

    int horizontalEdges = 0;
    int totalStrongEdges = 0;

    for (int y = 0; y < sobelX.rows; y++) {
        for (int x = 0; x < sobelX.cols; x++) {
            Vec3s sx = sobelX.at<Vec3s>(y, x);
            Vec3s sy = sobelY.at<Vec3s>(y, x);

            float gx = (sx[0] + sx[1] + sx[2]) / 3.0f;
            float gy = (sy[0] + sy[1] + sy[2]) / 3.0f;

            // Compute magnitude and orientation
            float mag = sqrt(gx * gx + gy * gy);

            if (mag > 30) {  // Only consider strong edges
                totalStrongEdges++;

                float angle = atan2(gy, gx) * 180.0f / CV_PI;
                if (angle < 0) angle += 360.0f;

                // Check if horizontal (0° ± 20° or 180° ± 20°)
                if ((angle < 20 || angle > 340) || (angle > 160 && angle < 200)) {
                    horizontalEdges++;
                }
            }
        }
    }

    float density = (totalStrongEdges > 0) ?
        (float)horizontalEdges / totalStrongEdges : 0.0f;

    printf("  Horizontal line density: %.3f\n", density);

    return density;
}


vector<float> findImageFeatures(string target_filename,
    vector<char*>& db_filenames,
    vector<vector<float>>& db_features) {
    /*
      Find feature vector for a specific image filename in database

      Arguments:
        target_filename - Full path or filename of target image
        db_filenames    - Vector of database image filenames
        db_features     - Vector of feature vectors for database images

      Returns:
        Feature vector if found, empty vector otherwise
    */


    // Extract just the filename from full path
    size_t lastSlash = target_filename.find_last_of("/\\");
    string filename = (lastSlash != string::npos) ?
        target_filename.substr(lastSlash + 1) :
        target_filename;

    // Search for filename in database
    for (int i = 0; i < db_filenames.size(); i++) {
        if (string(db_filenames[i]) == filename) {
            printf("Found target image in database at index %d\n", i);
            return db_features[i];
        }
    }

    printf("Warning: Target image not found in database\n");
    return vector<float>();  // Return empty vector
}


vector<float> extractCarFeatures(Mat& image,
    string image_filename,
    vector<char*>& dnn_filenames,
    vector<vector<float>>& dnn_features) {
    /*
      Extract all car-specific features combining DNN and custom features

      Arguments:
        image         - Input BGR color image (Mat)
        image_filename- Filename of the image
        dnn_filenames - Vector of filenames from ResNet CSV
        dnn_features  - Vector of ResNet18 embeddings

      Returns:
        Vector of 563 float values (512 DNN + 1 circles + 1 aspect + 16 lower + 32 metallic + 1 horizontal)
    */

    vector<float> features;

    printf("Extracting car features for: %s\n", image_filename.c_str());

    // 1. Get DNN embeddings from ResNet CSV (512 features)
    vector<float> dnn = findImageFeatures(image_filename, dnn_filenames, dnn_features);
    if (dnn.empty()) {
        printf("  Warning: DNN features not found, using zeros\n");
        dnn.resize(512, 0.0f);
    }
    features.insert(features.end(), dnn.begin(), dnn.end());
    printf("  DNN features: 512\n");

    // 2. Circle detection (1 feature)
    float circles = detectCircles(image);
    features.push_back(circles);

    // 3. Aspect ratio (1 feature)
    float aspect = getAspectRatio(image);
    features.push_back(aspect);

    // 4. Lower region texture (16 features)
    vector<float> lower = extractLowerRegionTexture(image);
    features.insert(features.end(), lower.begin(), lower.end());
    printf("  Lower region texture: 16\n");

    // 5. Metallic texture (32 features)
    vector<float> metallic = extractMetallicTexture(image);
    features.insert(features.end(), metallic.begin(), metallic.end());
    printf("  Metallic texture: 32\n");

    // 6. Horizontal line density (1 feature)
    float horizontal = getHorizontalLineDensity(image);
    features.push_back(horizontal);

    printf("  Total car features: %lu\n\n", features.size());

    return features;  // 563 total features
}


float computeSSD(vector<float>& features1, vector<float>& features2) {
    /*
      Compute Sum of Squared Differences between two feature vectors

      Arguments:
        features1 - First feature vector
        features2 - Second feature vector

      Returns:
        SSD distance (lower = more similar), -1.0 on error
    */

    float ssd = 0.0f;

    if (features1.size() != features2.size()) {
        printf("Error: Feature vectors have different sizes!\n");
        return -1.0f;
    }

    for (int i = 0; i < features1.size(); i++) {
        float diff = features1[i] - features2[i];
        ssd += diff * diff;
    }

    return ssd;
}


float computeHistogramIntersection(vector<float>& hist1, vector<float>& hist2) {
    /*
      Compute histogram intersection distance

      Arguments:
        hist1 - First normalized histogram
        hist2 - Second normalized histogram

      Returns:
        Distance as 1.0 - intersection (lower = more similar), -1.0 on error
    */

    float intersection = 0.0f;

    if (hist1.size() != hist2.size()) {
        printf("Error: Histogram sizes don't match!\n");
        return -1.0f;
    }

    // Sum of minimum values (both histograms are normalized)
    for (int i = 0; i < hist1.size(); i++) {
        intersection += min(hist1[i], hist2[i]);
    }

    // Convert to distance: lower intersection = higher distance
    return 1.0f - intersection;
}


float computeMultiHistogramIntersection(vector<float>& feat1, vector<float>& feat2) {
    /*
      Compute multi-histogram intersection distance for spatial histograms

      Arguments:
        feat1 - First feature vector [top_512, bottom_512]
        feat2 - Second feature vector [top_512, bottom_512]

      Returns:
        Weighted combined distance (0.5 top + 0.5 bottom), -1.0 on error
    */

    if (feat1.size() != 1024 || feat2.size() != 1024) {
        printf("Error: Multi-histogram features must have 1024 elements!\n");
        return -1.0f;
    }

    // Split into top and bottom histograms
    vector<float> top1(feat1.begin(), feat1.begin() + 512);
    vector<float> bottom1(feat1.begin() + 512, feat1.end());

    vector<float> top2(feat2.begin(), feat2.begin() + 512);
    vector<float> bottom2(feat2.begin() + 512, feat2.end());

    // Compute intersection for each region
    float top_intersection = 0.0f;
    float bottom_intersection = 0.0f;

    for (int i = 0; i < 512; i++) {
        top_intersection += min(top1[i], top2[i]);
        bottom_intersection += min(bottom1[i], bottom2[i]);
    }

    // Convert to distances
    float top_distance = 1.0f - top_intersection;
    float bottom_distance = 1.0f - bottom_intersection;

    // Combine with equal weights (0.5 each)
    float combined_distance = 0.5f * top_distance + 0.5f * bottom_distance;

    return combined_distance;
}


float computeColorTextureDistance(vector<float>& feat1, vector<float>& feat2) {
    /*
      Compute combined color and texture distance

      Arguments:
        feat1 - First feature vector [color_512, texture_16]
        feat2 - Second feature vector [color_512, texture_16]

      Returns:
        Weighted combined distance (0.5 color + 0.5 texture), -1.0 on error
    */

    if (feat1.size() != 528 || feat2.size() != 528) {
        printf("Error: Color+Texture features must have 528 elements!\n");
        return -1.0f;
    }

    // Split into color and texture
    vector<float> color1(feat1.begin(), feat1.begin() + 512);
    vector<float> texture1(feat1.begin() + 512, feat1.end());

    vector<float> color2(feat2.begin(), feat2.begin() + 512);
    vector<float> texture2(feat2.begin() + 512, feat2.end());

    // Compute histogram intersection for both
    float color_intersection = 0.0f;
    for (int i = 0; i < 512; i++) {
        color_intersection += min(color1[i], color2[i]);
    }

    float texture_intersection = 0.0f;
    for (int i = 0; i < 16; i++) {
        texture_intersection += min(texture1[i], texture2[i]);
    }

    // Convert to distances
    float color_distance = 1.0f - color_intersection;
    float texture_distance = 1.0f - texture_intersection;

    // Combine with equal weights
    float combined = 0.5f * color_distance + 0.5f * texture_distance;

    return combined;
}


float computeExtendedDistance(vector<float>& feat1, vector<float>& feat2) {
    /*
      Compute extended multi-feature distance

      Arguments:
        feat1 - First feature vector [color_512, magnitude_16, orientation_8, laws_80]
        feat2 - Second feature vector [color_512, magnitude_16, orientation_8, laws_80]

      Returns:
        Weighted combined distance (0.4 color + 0.2 mag + 0.2 orient + 0.2 laws), -1.0 on error
    */

    if (feat1.size() != 616 || feat2.size() != 616) {
        printf("Error: Extended features must have 616 elements!\n");
        return -1.0f;
    }

    // Split features
    vector<float> color1(feat1.begin(), feat1.begin() + 512);
    vector<float> mag1(feat1.begin() + 512, feat1.begin() + 528);
    vector<float> orient1(feat1.begin() + 528, feat1.begin() + 536);
    vector<float> laws1(feat1.begin() + 536, feat1.end());

    vector<float> color2(feat2.begin(), feat2.begin() + 512);
    vector<float> mag2(feat2.begin() + 512, feat2.begin() + 528);
    vector<float> orient2(feat2.begin() + 528, feat2.begin() + 536);
    vector<float> laws2(feat2.begin() + 536, feat2.end());

    // Compute histogram intersections
    float color_int = 0.0f;
    for (int i = 0; i < 512; i++) {
        color_int += min(color1[i], color2[i]);
    }

    float mag_int = 0.0f;
    for (int i = 0; i < 16; i++) {
        mag_int += min(mag1[i], mag2[i]);
    }

    float orient_int = 0.0f;
    for (int i = 0; i < 8; i++) {
        orient_int += min(orient1[i], orient2[i]);
    }

    float laws_int = 0.0f;
    for (int i = 0; i < 80; i++) {
        laws_int += min(laws1[i], laws2[i]);
    }

    //Normalizing laws_int
	laws_int = laws_int / 5.0f;  // Since we have 5 filters, we can divide by 5 to get an average intersection

    // Convert to distances
    float color_dist = 1.0f - color_int;
    float mag_dist = 1.0f - mag_int;
    float orient_dist = 1.0f - orient_int;
    float laws_dist = 1.0f - laws_int;

    // Weighted combination (color gets more weight)
    float combined = 0.4f * color_dist + 0.2f * mag_dist + 0.2f * orient_dist + 0.2f * laws_dist;

    return combined;
}


float computeCosineDistance(vector<float>& v1, vector<float>& v2) {
    /*
      Compute cosine distance between two vectors (angle-based similarity)

      Arguments:
        v1 - First vector (any dimension)
        v2 - Second vector (same dimension as v1)

      Returns:
        Cosine distance as 1.0 - cos(theta), where theta is angle between vectors
        Range: 0.0 (identical direction) to 2.0 (opposite direction), -1.0 on error
    */

    if (v1.size() != v2.size()) {
        printf("Error: Vectors must have same size for cosine distance\n");
        return -1.0f;
    }

    // Compute dot product
    float dot_product = 0.0f;
    for (int i = 0; i < v1.size(); i++) {
        dot_product += v1[i] * v2[i];
    }

    // Compute magnitudes (L2-norms)
    float mag1 = 0.0f;
    float mag2 = 0.0f;
    for (int i = 0; i < v1.size(); i++) {
        mag1 += v1[i] * v1[i];
        mag2 += v2[i] * v2[i];
    }
    mag1 = sqrt(mag1);
    mag2 = sqrt(mag2);

    // Avoid division by zero
    if (mag1 < 1e-10 || mag2 < 1e-10) {
        printf("Warning: Zero magnitude vector\n");
        return 1.0f;
    }

    // Compute cosine similarity: cos(theta) = dot_product / (mag1 * mag2)
    float cosine_similarity = dot_product / (mag1 * mag2);

    // Clamp to [-1, 1] to handle floating point errors
    if (cosine_similarity > 1.0f) cosine_similarity = 1.0f;
    if (cosine_similarity < -1.0f) cosine_similarity = -1.0f;

    // Convert to distance: d = 1 - cos(theta)
    float cosine_distance = 1.0f - cosine_similarity;

    return cosine_distance;
}


float computeCarDistance(vector<float>& feat1, vector<float>& feat2) {
    /*
      Compute custom car retrieval distance combining DNN and domain features

      Arguments:
        feat1 - First car feature vector [DNN_512, circles_1, aspect_1, lower_16, metallic_32, horiz_1]
        feat2 - Second car feature vector (same structure)

      Returns:
        Weighted combined distance (0.35 DNN + 0.20 circles + 0.15 aspect +
                                    0.15 lower + 0.10 metallic + 0.05 horizontal), -1.0 on error
    */

    if (feat1.size() != 563 || feat2.size() != 563) {
        printf("Error: Car features must have 563 elements! Got %lu and %lu\n",
            feat1.size(), feat2.size());
        return -1.0f;
    }

    // Split features
    vector<float> dnn1(feat1.begin(), feat1.begin() + 512);
    float circle1 = feat1[512];
    float aspect1 = feat1[513];
    vector<float> lower1(feat1.begin() + 514, feat1.begin() + 530);
    vector<float> metallic1(feat1.begin() + 530, feat1.begin() + 562);
    float horiz1 = feat1[562];

    vector<float> dnn2(feat2.begin(), feat2.begin() + 512);
    float circle2 = feat2[512];
    float aspect2 = feat2[513];
    vector<float> lower2(feat2.begin() + 514, feat2.begin() + 530);
    vector<float> metallic2(feat2.begin() + 530, feat2.begin() + 562);
    float horiz2 = feat2[562];

    // 1. DNN cosine distance (35%)
    float dnn_dist = computeCosineDistance(dnn1, dnn2);

    // 2. Circle detection distance (20%)
    float circle_dist = abs(circle1 - circle2);

    // 3. Aspect ratio distance (15%)
    float aspect_dist = abs(aspect1 - aspect2) / 3.0f;  // Normalize (typical range 0-3)
    if (aspect_dist > 1.0f) aspect_dist = 1.0f;

    // 4. Lower texture histogram intersection (15%)
    float lower_int = 0.0f;
    for (int i = 0; i < 16; i++) {
        lower_int += min(lower1[i], lower2[i]);
    }
    float lower_dist = 1.0f - lower_int;

    // 5. Metallic texture histogram intersection (10%)
    float metallic_int = 0.0f;
    for (int i = 0; i < 32; i++) {
        metallic_int += min(metallic1[i], metallic2[i]);
    }
    metallic_int = metallic_int / 2.0f;
    float metallic_dist = 1.0f - metallic_int;

    // 6. Horizontal line density distance (5%)
    float horiz_dist = abs(horiz1 - horiz2);

    // Weighted combination
    float combined = 0.35f * dnn_dist +
        0.20f * circle_dist +
        0.15f * aspect_dist +
        0.15f * lower_dist +
        0.10f * metallic_dist +
        0.05f * horiz_dist;

    return combined;
}


/*
  Structure to hold image filename and distance
*/
struct ImageMatch {
    string filename;
    float distance;

    ImageMatch(string fname, float dist) : filename(fname), distance(dist) {}
};

/*
  Comparison function for sorting
*/
bool compareByDistance(const ImageMatch& a, const ImageMatch& b) {
    return a.distance < b.distance;
}


int main(int argc, char* argv[]) {
    char csv_filename[256];
    char target_filename[256];
    int N = 3;
    string distance_metric = "ssd";  // Default

    if (argc < 3) {
        printf("Usage: %s <target_image> <features_csv> [N] [distance_metric]\n", argv[0]);
        printf("  N: number of matches (default: 3)\n");
        printf("  distance_metric: 'ssd', 'histogram', 'multi', 'texture', 'extended', 'embedding', or 'car'\n");
        return -1;
    }

    strcpy(target_filename, argv[1]);
    strcpy(csv_filename, argv[2]);
    if (argc >= 4) {
        N = atoi(argv[3]);
    }
    if (argc >= 5) {
        distance_metric = argv[4];
    }

    printf("Target image: %s\n", target_filename);
    printf("Features database: %s\n", csv_filename);
    printf("Distance metric: %s\n", distance_metric.c_str());
    printf("Returning top %d matches\n\n", N);

    // Read target image
    Mat target_image = imread(target_filename);
    if (target_image.empty()) {
        printf("Error: Cannot read target image\n");
        return -1;
    }

    // Step 1: Read database FIRST
    vector<char*> db_filenames;
    vector<vector<float>> db_features;

    if (read_image_data_csv(csv_filename, db_filenames, db_features, 0) != 0) {
        printf("Error: Failed to read CSV\n");
        return -1;
    }
    printf("Loaded %lu images from database\n\n", db_filenames.size());


    // For car mode, we need ResNet features too
    vector<char*> resnet_filenames;
    vector<vector<float>> resnet_features;
    bool has_resnet = false;

    if (distance_metric == "car") {
        // Load ResNet features for DNN embeddings
        char resnet_csv[256] = "ResNet18_olym.csv";  // Adjust path if needed
        if (read_image_data_csv(resnet_csv, resnet_filenames, resnet_features, 0) == 0) {
            printf("Loaded ResNet features: %lu images\n\n", resnet_filenames.size());
            has_resnet = true;
        }
        else {
            printf("Warning: Could not load ResNet features. Car detection may be less accurate.\n\n");
        }
    }



    // Step 2: Get target features
    vector<float> target_features;

    if (distance_metric == "embedding") {
        // For embeddings, find target features in CSV
        target_features = findImageFeatures(target_filename, db_filenames, db_features);
        if (target_features.empty()) {
            printf("Error: Could not find target image features\n");
            return -1;
        }
    }
    else if (distance_metric == "car") {
        // For car detection, compute custom features
        Mat target_image = imread(target_filename);
        if (target_image.empty()) {
            printf("Error: Cannot read target image\n");
            return -1;
        }

        if (has_resnet) {
            target_features = extractCarFeatures(target_image, target_filename,
                resnet_filenames, resnet_features);
        }
        else {
            printf("Error: ResNet features required for car detection\n");
            return -1;
        }

        // Now compute features for all database images
        printf("Computing car features for all database images...\n");
        printf("This may take a few minutes...\n\n");

        // Clear existing features and recompute with car features
        vector<vector<float>> car_features;
        for (int i = 0; i < db_filenames.size(); i++) {
            // Build full path
            string img_path = string("C:\\Users\\ahiva\\source\\repos\\Project2\\Project2\\olympus\\olympus\\") + string(db_filenames[i]);

            Mat img = imread(img_path);
            if (img.empty()) {
                printf("Warning: Could not read %s\n", db_filenames[i]);
                // Create empty feature vector
                car_features.push_back(vector<float>(563, 0.0f));
                continue;
            }

            vector<float> feats = extractCarFeatures(img, string(db_filenames[i]),
                resnet_filenames, resnet_features);
            car_features.push_back(feats);

            if ((i + 1) % 50 == 0) {
                printf("Processed %d/%lu images...\n", i + 1, db_filenames.size());
            }
        }

        printf("Finished computing car features!\n\n");

        // Replace db_features with car_features
        db_features = car_features;
    }
    else {
        // For other methods, read image and compute features
        Mat target_image = imread(target_filename);
        if (target_image.empty()) {
            printf("Error: Cannot read target image\n");
            return -1;
        }

        if (distance_metric == "histogram") {
            target_features = extractRGBHistogram(target_image);
        }
        else if (distance_metric == "multi") {
            target_features = extractMultiHistogram(target_image);
        }
        else if (distance_metric == "texture") {
            target_features = extractColorAndTexture(target_image);
        }
        else if (distance_metric == "extended") {
            target_features = extractAllFeatures(target_image);
        }
        else {
            target_features = extract7x7Features(target_image);
        }
    }

    printf("Extracted %lu features from target\n", target_features.size());

    // Compute distances
    vector<ImageMatch> matches;

    for (int i = 0; i < db_filenames.size(); i++) {
        float distance;

        if (distance_metric == "embedding") {
            distance = computeCosineDistance(target_features, db_features[i]);
        }
        else if (distance_metric == "car") {
            distance = computeCarDistance(target_features, db_features[i]);
        }
        else if (distance_metric == "histogram") {
            distance = computeHistogramIntersection(target_features, db_features[i]);
        }
        else if (distance_metric == "multi") {
            distance = computeMultiHistogramIntersection(target_features, db_features[i]);
        }
        else if (distance_metric == "texture") {
            distance = computeColorTextureDistance(target_features, db_features[i]);
        }
        else if (distance_metric == "extended") {
            distance = computeExtendedDistance(target_features, db_features[i]);
        }
        else {
            distance = computeSSD(target_features, db_features[i]);
        }

        matches.push_back(ImageMatch(string(db_filenames[i]), distance));
    }

    // Sort by distance
    sort(matches.begin(), matches.end(), compareByDistance);

    // Display results
    printf("Top %d matches:\n", N);
    printf("%-5s %-30s %s\n", "Rank", "Filename", "Distance");
    printf("-----------------------------------------------------\n");

    for (int i = 0; i < N && i < matches.size(); i++) {
        printf("%-5d %-30s %.6f\n",
            i + 1,
            matches[i].filename.c_str(),
            matches[i].distance);
    }

    // Cleanup
    for (int i = 0; i < db_filenames.size(); i++) {
        delete[] db_filenames[i];
    }

    return 0;
}