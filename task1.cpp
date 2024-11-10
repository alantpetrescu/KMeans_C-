#include <random>
#include <iostream>
#include <vector>
#include <utility>
#include <cfloat>
#include <chrono>

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

void assign_cluster_indexes(const VLPoints &points, const VLPoints &centroids, VInts &cluster_indexes) {
    int n = points.size();
    int k = centroids.size();

    for(int i = 0; i < n; i++) {
        double min_dist = DBL_MAX;
        int min_centroid_index = 0;

        for(int j = 0; j < k; j++) {
            double dist = calculate_distance_between_points(points[i], centroids[j]);
            if(dist < min_dist) {
                min_dist = dist;
                min_centroid_index = j;
            }
        }

        cluster_indexes[i] = min_centroid_index;
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