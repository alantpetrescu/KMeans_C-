#include <random>
#include <iostream>
#include <vector>
#include <utility>
#include <cfloat>
#include <chrono>
#include <thread>
#include <immintrin.h>
#include <float.h>

#define NR_POINTS 1'000'000
#define POINT_MAX_SIZE 1'000'000
#define NR_CLUSTERS 5
#define NR_CENTROID_UPDATE_STEPS 50

#define Point pair<float, float>
#define LPoint pair<double, double>
#define VPoints vector<Point>
#define VLPoints vector<LPoint>
#define VInts vector<int>

using namespace std;

VLPoints points(NR_POINTS);
VLPoints centroids(NR_CLUSTERS);
VInts cluster_indexes(NR_POINTS);

inline long long choose_random_number(int st, int dr) {
    return rand() * rand() % (dr - st) + st;
}

void choose_random_points(VLPoints &points) {
    int n = points.size();

    for(int i = 0; i < n; i++) {
        points[i] = {(double)choose_random_number(0, POINT_MAX_SIZE), (double)choose_random_number(0, POINT_MAX_SIZE)};
    }
}

inline double calculate_distance_between_points(const LPoint& a, const LPoint& b) {
    return (b.first - a.first) * (b.first - a.first) + 
           (b.second - a.second) * (b.second - a.second);
}

inline double my_min(double &a, double &b) {
    int mask = int(b - a);

    return a - ((b - a) * (mask >> 31));
}

pair<double, int> take_minimum_of_m256d(__m256d &v) {
    __m256d index = _mm256_set_pd(1.0, 2.0, 3.0, 4.0);

    __m256d temp = _mm256_permute4x64_pd(v, _MM_SHUFFLE(2, 3, 0, 1));
    __m256d index_temp = _mm256_permute4x64_pd(index, _MM_SHUFFLE(2, 3, 0, 1));
    __m256d min1 = _mm256_min_pd(v, temp);
    __m256d min1_index = _mm256_blendv_pd(index, index_temp, _mm256_cmp_pd(v, temp, _CMP_GT_OS));

    temp = _mm256_permute4x64_pd(min1, _MM_SHUFFLE(1,0,3,2));
    index_temp = _mm256_permute4x64_pd(min1_index, _MM_SHUFFLE(1,0,3,2));
    __m256d min = _mm256_min_pd(min1, temp);
    __m256d min_index = _mm256_blendv_pd(min1_index, index_temp, _mm256_cmp_pd(min1, temp, _CMP_GT_OS));

    return {_mm256_cvtsd_f64(min), (int)_mm256_cvtsd_f64(min_index)};
}

void assign_cluster_indexes_chunk(int start, int end, const VLPoints &points, 
                                    const VLPoints &centroids, VInts &cluster_indexes) {
    int n = points.size();
    int k = centroids.size();
    int div = k / 4;

    for(int i = start; i <= end; i++) {
        double min_dist = DBL_MAX;
        int min_index = -1;

        for(int j = 0; j < div; j++) {
            __m256d points_x = _mm256_set_pd(points[i].first, points[i].first, points[i].first, points[i].first);
            __m256d points_y = _mm256_set_pd(points[i].second, points[i].second, points[i].second, points[i].second);
            __m256d centroids_x = _mm256_set_pd(centroids[0].first, centroids[1].first, centroids[2].first, centroids[3].first);
            __m256d centroids_y = _mm256_set_pd(centroids[0].second, centroids[1].second, centroids[2].second, centroids[3].second);

            __m256d dist_x = _mm256_sub_pd(points_x, centroids_x);
            __m256d dist_x_2 = _mm256_mul_pd(dist_x, dist_x);
            __m256d dist_y = _mm256_sub_pd(points_y, centroids_y);
            __m256d dist_y_2 = _mm256_mul_pd(dist_y, dist_y);
            __m256d new_dist = _mm256_add_pd(dist_x_2, dist_y_2);

            pair<double, int> ans = take_minimum_of_m256d(new_dist);
            min_dist = ans.first;
            min_index = ans.second;
        }

        for(int j = div * 4; j < k; j++) {
            double new_dist = calculate_distance_between_points(points[i], centroids[j]);
            min_dist = my_min(min_dist, new_dist);
            bool mask = -(new_dist == min_dist);
            min_index = (mask & j) | (~mask & min_index);
        }

        cluster_indexes[i] = min_index;
    }
}

void assign_cluster_indexes(const VLPoints &points, const VLPoints &centroids, VInts &cluster_indexes) {
    int n = points.size();
    int k = centroids.size();

    vector<thread> threads;

    int threads_count = thread::hardware_concurrency();
    int chunk_size = n / threads_count;

    for(int thread_id = 0; thread_id < threads_count - 1; thread_id++) {
        int start = thread_id * chunk_size;
        int end = start + chunk_size - 1;

        threads.emplace_back(assign_cluster_indexes_chunk, start, end, ref(points), ref(centroids), ref(cluster_indexes));
    }

    int start = (threads_count - 1) * chunk_size;
    int end = n - 1;
    threads.emplace_back(assign_cluster_indexes_chunk, start, end, ref(points), ref(centroids), ref(cluster_indexes));

    for(thread& t : threads) {
        if (t.joinable())
            t.join();
    }
}

inline void add_two_points(LPoint &a, const LPoint &b) {
    a.first += b.first;
    a.second += b.second;
}

void display_centroids(const VLPoints &centroids) {
    cout << "Cluster centroids: [";

    for(int i = 0; i < NR_CLUSTERS - 1; i++) {
        cout << "(" << centroids[i].first << "," << centroids[i].second << "), ";
    }

    cout << "(" << centroids[NR_CLUSTERS - 1].first << "," << centroids[NR_CLUSTERS - 1].second << ")]\n";
}

void calculate_new_centroids(const VLPoints &points, VLPoints &centroids, const VInts &cluster_indexes) {
    int n = points.size();
    int k = centroids.size();
    VLPoints new_centroids(k);
    VInts centroid_index_counter(k);

    for(int i = 0; i < n; i++) {
        add_two_points(new_centroids[cluster_indexes[i]], points[i]);
        centroid_index_counter[cluster_indexes[i]]++;
    }

    for(int i = 0; i < k; i++) {
        new_centroids[i].first /= centroid_index_counter[i];
        new_centroids[i].second /= centroid_index_counter[i];

        centroids[i].first = new_centroids[i].first;
        centroids[i].second = new_centroids[i].second;
    }
}

void display_time(long long microseconds_count, long long max_power = 1e6) {
    long long copy_microseconds_count = microseconds_count;
    long long power = 1, after = 0;
    
    while(power < max_power) {
        after = power * (copy_microseconds_count % 10) + after;
        copy_microseconds_count /= 10;
        power *= 10;
    }

    cout << copy_microseconds_count << "." << after << " seconds\n";
}

int main()
{
    auto start = chrono::high_resolution_clock::now();
    choose_random_points(points);
    choose_random_points(centroids);

    for(int i = 0; i < NR_CENTROID_UPDATE_STEPS; i++) {
        assign_cluster_indexes(points, centroids, cluster_indexes);
        calculate_new_centroids(points, centroids, cluster_indexes);
    }

    display_centroids(centroids);

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "Total time elapsed(microseconds): ";
    display_time(duration.count());

    return 0;
}