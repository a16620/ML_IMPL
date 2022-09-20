/*

1개 샘플 선택


*/


#include <random>
#include <vector>
#include <iterator>
#include <algorithm>
#include <fstream>
#include <iostream>
using namespace std;

struct xy {
    xy() : x(0), y(0) {}
    xy(float x_, float y_) : x(x_), y(y_) {}

    float x, y;
};

inline float sqrtDist(const xy& p, const xy& p2) {
    return powf(p.x - p2.x, 2) + powf(p.y - p2.y, 2);
}

vector<xy> get_sample_data();
vector<xy> get_sample_data(const xy& center);
void save_sample_data(vector<xy>& dataset, const vector<int>& gid);

int argmin(const vector<float>& v);
int argmin(const vector<float>& v, const vector<int>& gid, int g_target);

int main() {
    const size_t k = 7;

    auto dataset = vector<xy>();
    const auto points = vector<xy>();


    {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(-200.0f, 200.0f);

        for (int i = 0; i < 10; i++) {
            xy p(dis(gen), dis(gen));
            auto s = get_sample_data(p);
            copy(s.begin(), s.end(), back_inserter(dataset));
        }
    }
    const auto data_cnt = dataset.size();

    vector<int> group_id(data_cnt, -1);
    vector<xy> group_center(k);

    vector<float> dist;
    dist.reserve(data_cnt);
    {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, data_cnt - 1);
        const int inital_select_idx = dis(gen);

        const auto& g_center = (group_center[0] = dataset[inital_select_idx]);
        transform(dataset.begin(), dataset.end(), back_inserter(dist), [&g_center](const xy& data) {
            return sqrtDist(data, g_center);
            });
    }


    for (int i = 1; i < k; i++) {
        int next = argmin(dist, group_id, -1);

        group_center[i] = dataset[next];
        group_id[next] = i;

        for (int j = 0; j < data_cnt; j++) {
            dist[j] += sqrtDist(group_center[i], dataset[j]);
        }
    }
    dist.clear();

    fill_n(back_inserter(dist), k, 0);
    vector<size_t> group_cnt(k, 1);

    //1차 배정
    for (int i = 0; i < data_cnt; i++) {
        if (group_id[i] != -1)
            continue;

        const auto& p_cur = dataset[i];
        transform(group_center.begin(), group_center.end(), dist.begin(), [&p_cur](const xy& p) {
            return sqrtDist(p_cur, p);
            });
        const auto gid = argmin(dist);
        group_id[i] = gid;
        group_center[gid].x += p_cur.x;
        group_center[gid].y += p_cur.y;
        group_cnt[gid]++;
    }

    for (int i = 0; i < k; i++) {
        group_center[i].x /= group_cnt[i];
        group_center[i].y /= group_cnt[i];
    }


    int cnt = 50000;
    while (cnt-- > 0) {
        for (int i = 0; i < data_cnt; i++) {
            const auto& p_cur = dataset[i];
            transform(group_center.begin(), group_center.end(), dist.begin(), [&p_cur](const xy& p) {
                return sqrtDist(p_cur, p);
                });

            const auto g_from = group_id[i], g_new = argmin(dist);
            if (g_from == g_new)
                continue;


            auto& from_center = group_center[g_from];
            auto& from_cnt = group_cnt[g_from];

            if (from_cnt == 1) {
                continue;
            }

            group_id[i] = g_new;

            from_center.x *= from_cnt;
            from_center.y *= from_cnt;

            from_center.x -= p_cur.x;
            from_center.y -= p_cur.y;
            from_cnt--;

            from_center.x /= from_cnt;
            from_center.y /= from_cnt;

            auto& new_center = group_center[g_new];
            auto& new_cnt = group_cnt[g_new];

            new_center.x *= new_cnt;
            new_center.y *= new_cnt;

            new_center.x += p_cur.x;
            new_center.y += p_cur.y;
            new_cnt++;

            new_center.x /= new_cnt;
            new_center.y /= new_cnt;
        }
    }

    save_sample_data(dataset, group_id);

    for (int i = 0; i < k; i++) {
        cout << i << ":(" << group_center[i].x << ',' << group_center[i].y << ")\n";
    }

    return 0;
}

vector<xy> get_sample_data(const xy& center) {
    const auto batch_cnt = 50;
    vector<xy> dataset;
    dataset.reserve(batch_cnt);

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> ndis(0, 50);

    for (int i = 0; i < batch_cnt; i++) {
        dataset.emplace_back(center.x + ndis(gen), center.y + ndis(gen));
    }

    return dataset;
}

void save_sample_data(vector<xy>& dataset, const vector<int>& gid) {
    ofstream fs("out.csv");

    fs << "x" << ',' << "y" << ',' << "group" << '\n';
    const auto end = dataset.size();
    for (int i = 0; i < end; i++) {
        fs << dataset[i].x << ',' << dataset[i].y << ',' << gid[i] << '\n';
    }

    fs.close();
}


int argmin(const vector<float>& v) {
    int max = 0;
    const auto end = v.size();
    for (int i = 1; i < end; i++) {
        if (v[i] < v[max]) {
            max = i;
        }
    }
    return max;
}

int argmin(const vector<float>& v, const vector<int>& gid, int g_target) {
    int i = 0, min_idx = -1;
    const auto end = v.size();

    while (i < end) {
        if (gid[i] == g_target) {
            min_idx = i++;
            break;
        }
        i++;
    }

    while (i < end) {
        if (gid[i] != g_target) {
            i++;
            continue;
        }
        if (v[i] < v[min_idx]) {
            min_idx = i;
        }
        i++;
    }

    return min_idx;
}