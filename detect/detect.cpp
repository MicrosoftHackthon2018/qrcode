#include <algorithm>
#include <iostream>
#include <vector>

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#include <opencv2/opencv.hpp>

using namespace std;

int dist2(cv::Point &p1, cv::Point &p2)
{
    int dx = p1.x - p2.x, dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

void least_squre(vector<cv::Point> &p, size_t max_idx, size_t i0, size_t i1, float *a, float *b, int *axb)
{
    const size_t n = (i1 + max_idx + 1 - i0) % max_idx;
    float x = 0.0f, y = 0.0f, xx = 0.0f, xy = 0.0f;

    bool axb_ = abs(p[i0].x - p[i1].x) > abs(p[i0].y - p[i1].y);
    float xp, yp;
    for (size_t i = i0;;) {
        xp = axb_ ? p[i].x : p[i].y;
        yp = axb_ ? p[i].y : p[i].x;
        x += xp;
        y += yp;
        xx += xp * xp;
        xy += xp * yp;

        if (i == i1)
            break;
        ++i;
        i %= max_idx;
    }

    const float m = n * xx - x * x;
    *a = (n * xy - x * y) / m;
    *b = (xx * y - x * xy) / m;
    *axb = axb_;
}

void plot_line(cv::Mat &img, int width, int height, float a, float b, int axb)
{
    if (axb)
        cv::line(img, cv::Point(0, (int)b), cv::Point(width, (int)(a * width + b)), cv::Scalar(255, 0, 0));
    else
        cv::line(img, cv::Point((int)b, 0), cv::Point((int)(a * height + b), height), cv::Scalar(0, 0, 255));
}

bool same_line(cv::Point &center, float dist,
    float a1, float b1, int axb1,
    float a2, float b2, int axb2)
{
#define MAX_A_DIFF  0.1f
    if (axb1 == axb2) {
        if (abs(a1 - a2) < MAX_A_DIFF && abs((a1 - a2) * (axb1 ? center.x : center.y) + b1 - b2) < dist / 4)
            return true;
        return false;
    }
    if (abs(a2) < MAX_A_DIFF)
        return false;
    a2 = 1.0f / a2;
    b2 = -b2 * a2;
    if (abs(a1 - a2) < MAX_A_DIFF && abs((a1 - a2) * (axb1 ? center.x : center.y) + b1 - b2) < dist / 4)
        return true;
    return false;
}

int has_same_line(cv::Point &center, int dist,
    vector<float> &a1, vector<float> &b1, vector<int> &axb1,
    vector<float> &a2, vector<float> &b2, vector<int> &axb2)
{
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            if (same_line(center, dist, a1[i], b1[i], axb1[i], a2[j], b2[j], axb2[j])) {
                //printf("1: %d\t%d\n", i, j);
                int i2 = i + 2, j2 = (j + 2) % 4;
                if (same_line(center, dist, a1[i2], b1[i2], axb1[i2], a2[j2], b2[j2], axb2[j2])) {
                    //printf("2: %d\t%d\n", i, j);
                    return i;
                }
            }
        }
    }
    return -1;
}

int main(void)
{
    int threshold = 60;
    bool qr_found = false;
    int tested = 0;

    for (int img_num = 4200; img_num <= 4804; ) {
        char img_name[50] = "";
        sprintf_s(img_name, "D:\\git\\12345\\4\\%d.jpg", img_num);
        FILE *fp = 0;
        fopen_s(&fp, img_name, "r");
        if (!fp) {
            img_num++;
            continue;
        }
        fclose(fp);

        if (!qr_found) {
            if (++tested >= 3) {
                tested = 0;
                img_num++;
            }
            switch (threshold) {
            case 60: threshold = 100; break;
            case 100: threshold = 140; break;
            default: threshold = 60; break;
            }
        } else {
            tested = 0;
        }
        qr_found = false;

        printf("%s\t%d\n", img_name, threshold);
        cv::Mat src = cv::imread(img_name), gray;
        const int width = src.cols, height = src.rows;

        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);

        cv::threshold(gray, gray, threshold, 255, cv::THRESH_BINARY);
        cv::imshow("threshold", gray);

        cv::Canny(gray, gray, 50, 200);
        cv::imshow("edge", gray);

        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarchy;
        cv::findContours(gray, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

        // Find corner block of QR-code
        vector<size_t> found;
        for (size_t i = 0; i < hierarchy.size(); i++) {
            int c = 0, k = i;
            while ((k = hierarchy[k][2]) != -1)
                c++;
            if (c >= 5)
                found.push_back(i);
        }

        const size_t found_n = found.size();
        if (found_n == 0)
            continue;

        // center of each block
        vector<cv::Point> center = vector<cv::Point>(found_n, cv::Point(-1, -1));

        // y = ax + b, or, x = ay + b
        vector<vector<float>> a = vector<vector<float>>(found_n, vector<float>(4, 0.0f));
        vector<vector<float>> b = vector<vector<float>>(found_n, vector<float>(4, 0.0f));
        vector<vector<int>> axb = vector<vector<int>>(found_n, vector<int>(4, 1));

        for (size_t i = 0; i < found_n; i++) {
            vector<cv::Point> &contour = contours[found[i]];
            cv::drawContours(src, contours, found[i], cv::Scalar(0, 255, 0), 3);

            size_t n = contour.size();
            center[i].x = (contour[0].x + contour[n / 4].x + contour[n / 2].x + contour[n * 3 / 4].x) / 4;
            center[i].y = (contour[0].y + contour[n / 4].y + contour[n / 2].y + contour[n * 3 / 4].y) / 4;

            vector<int> d = vector<int>(n);
            for (size_t j = 0; j < n; j++)
                d[j] = dist2(center[i], contour[j]);

            const size_t CORNER_WIDTH = n < 8 ? 1 : n / 8;
            size_t count = 0;
            size_t corner_idx[4];
            for (size_t j = 0; j < n; j++) {
                if (d[j] > d[(j + CORNER_WIDTH) % n] && d[j] > d[(j + n - CORNER_WIDTH) % n]) {
                    cv::circle(src, contour[j], 5, cv::Scalar(0, 0, 255));
                    cv::circle(src, contour[(j + CORNER_WIDTH) % n], 5, cv::Scalar(255, 0, 0));
                    cv::circle(src, contour[(j + n - CORNER_WIDTH) % n], 5, cv::Scalar(0, 255, 0));
                    corner_idx[count % 4] = j;
                    j += CORNER_WIDTH - 1;
                    if (++count > 4)
                        break;
                }
            }
            if (count < 4) {
                center[i].x = -1;
                break;
            }

            least_squre(contour, n, (corner_idx[0] + CORNER_WIDTH) % n, corner_idx[1], &a[i][0], &b[i][0], &axb[i][0]);
            least_squre(contour, n, (corner_idx[1] + CORNER_WIDTH) % n, corner_idx[2], &a[i][1], &b[i][1], &axb[i][1]);
            least_squre(contour, n, (corner_idx[2] + CORNER_WIDTH) % n, corner_idx[3], &a[i][2], &b[i][2], &axb[i][2]);
            least_squre(contour, n, (corner_idx[3] + CORNER_WIDTH) % n, corner_idx[0], &a[i][3], &b[i][3], &axb[i][3]);

            cv::Mat c_src;
            src.copyTo(c_src);
            for (int j = 0; j < 4; j++) {
                plot_line(c_src, width, height, a[i][j], b[i][j], axb[i][j]);
            }
        }

        // match
        vector<int> flag = vector<int>(found_n, -1);
        int qr_idx[3] = { -1 };
        for (size_t i = 0; i < found_n; i++) {
            if (center[i].x == -1)
                continue;
            for (size_t j = i + 1; j < found_n; j++) {
                if (center[j].y == -1)
                    continue;
#define CHONG_HE_DIST2   400
                const int dist2_ = dist2(center[i], center[j]);
                if (dist2_ < CHONG_HE_DIST2)
                    continue;
                int res = has_same_line(center[i], sqrt(dist2_), a[i], b[i], axb[i], a[j], b[j], axb[j]);
                printf("i: %d\tj: %d\tres: %d\n", i, j, res);
                if (res == -1)
                    continue;
                if (flag[i] == -1) {
                    flag[i] = j;
                } else {
                    qr_found = true;
                    qr_idx[0] = i;
                    qr_idx[1] = flag[i];
                    qr_idx[2] = j;
                    break;
                }
                if (flag[j] == -1) {
                    flag[j] = i;
                } else {
                    qr_found = true;
                    qr_idx[0] = j;
                    qr_idx[1] = flag[j];
                    qr_idx[2] = i;
                    break;
                }
            }
            if (qr_found)
                break;
        }

        // qr_idx[0] is the middle block, which usually on the left-up side.

        if (qr_found) {
            img_num++;
            cv::line(src, center[qr_idx[0]], center[qr_idx[1]], cv::Scalar(255, 0, 0), 3);
            cv::line(src, center[qr_idx[0]], center[qr_idx[2]], cv::Scalar(0, 0, 255), 3);
        }

        cv::imshow("result", src);

        cv::waitKey();
    }

    return 0;
}