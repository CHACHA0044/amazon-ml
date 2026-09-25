#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <omp.h>
#include "lgbm_model.h"

// Problem & Matching Constants
constexpr float ANCHOR_THRESH = 0.85f;
constexpr size_t MAX_MATCHES_PER_S1 = 11;
constexpr size_t MAX_CAND_PER_QUERY = 120;

// Token normalization and stopword sets
static const std::unordered_set<std::string> NAME_STOPWORDS = {
    "ltd", "pvt", "inc", "llc", "corp", "co", "limited", "private", "corporation",
    "company", "services", "solutions", "technologies", "technology", "enterprises",
    "associates", "group", "partners", "international", "global", "trading", "brothers",
    "ventures", "industries", "consultancy", "consultants", "care", "and", "the", "of",
    "in", "for", "center", "centre", "hub", "studio", "sarl", "sas", "llp", "pllc"
};

static const std::unordered_set<std::string> ADDR_STOPWORDS = {
    "road", "rd", "street", "st", "lane", "ln", "avenue", "ave", "boulevard", "blvd",
    "drive", "dr", "highway", "hwy", "nagar", "colony", "block", "sector", "phase",
    "plot", "floor", "near", "opp", "opposite", "behind", "beside", "building", "bldg",
    "house", "no", "flat", "room", "apartment", "apt", "complex", "plaza", "market",
    "bazaar", "cross", "main", "layout", "extension", "ext", "dist", "district",
    "state", "city", "town", "village", "taluk", "tehsil", "post", "po", "rue", "allée"
};

inline bool is_alnum(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

inline bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

inline char to_lower(char c) {
    return (c >= 'A' && c <= 'Z') ? (c + 32) : c;
}

std::string clean_str(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    bool prev_space = true;
    for (char c : input) {
        if (is_alnum(c)) {
            out.push_back(to_lower(c));
            prev_space = false;
        } else if (!prev_space) {
            out.push_back(' ');
            prev_space = true;
        }
    }
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

std::vector<std::string> tokenize(const std::string& str, const std::unordered_set<std::string>* stopwords = nullptr) {
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < str.size()) {
        while (i < str.size() && str[i] == ' ') i++;
        if (i >= str.size()) break;
        size_t start = i;
        while (i < str.size() && str[i] != ' ') i++;
        std::string tok = str.substr(start, i - start);
        if (tok.size() >= 2 && (!stopwords || stopwords->find(tok) == stopwords->end())) {
            tokens.push_back(tok);
        }
    }
    return tokens;
}

std::vector<std::string> extract_numbers(const std::string& str) {
    std::vector<std::string> nums;
    size_t i = 0;
    while (i < str.size()) {
        while (i < str.size() && !is_digit(str[i])) i++;
        if (i >= str.size()) break;
        size_t start = i;
        while (i < str.size() && is_digit(str[i])) i++;
        nums.push_back(str.substr(start, i - start));
    }
    return nums;
}

std::vector<uint64_t> get_char_ngrams(const std::string& str, int n) {
    std::vector<uint64_t> ngrams;
    if (str.size() < (size_t)n) return ngrams;
    ngrams.reserve(str.size() - n + 1);
    for (size_t i = 0; i <= str.size() - n; ++i) {
        uint64_t h = 0;
        for (int k = 0; k < n; ++k) {
            h = (h * 131) + (uint8_t)str[i + k];
        }
        ngrams.push_back(h);
    }
    std::sort(ngrams.begin(), ngrams.end());
    ngrams.erase(std::unique(ngrams.begin(), ngrams.end()), ngrams.end());
    return ngrams;
}

inline float dice_sim(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    if (a.empty() && b.empty()) return 0.0f;
    size_t i = 0, j = 0, inter = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) { inter++; i++; j++; }
        else if (a[i] < b[j]) { i++; }
        else { j++; }
    }
    return (2.0f * inter) / (float)(a.size() + b.size());
}

// Compact Record for Candidates (Minimal RAM)
struct Record {
    std::string id;
    int src;
    std::string country;
    std::string name_norm;
    std::string addr_norm;
    std::vector<std::string> name_toks;
    std::vector<std::string> addr_toks;
    std::vector<std::string> nums;
    bool is_us;
    bool addr_missing;
};

// Full Record for Queries (Precomputed N-Grams)
struct QueryRecord {
    Record base;
    std::vector<uint64_t> name_c2;
    std::vector<uint64_t> name_c3;
    std::vector<uint64_t> addr_c2;
    std::vector<uint64_t> addr_c3;
};

// Lightweight Inverted Index
struct CandidatePool {
    std::vector<Record> cands;
    std::unordered_map<std::string, std::vector<uint32_t>> idx_name_exact;
    std::unordered_map<std::string, std::vector<uint32_t>> idx_addr_exact;
    std::unordered_map<std::string, std::vector<uint32_t>> idx_name_tok;
    std::unordered_map<std::string, std::vector<uint32_t>> idx_num;

    void build(std::vector<Record>&& records) {
        cands = std::move(records);
        for (uint32_t i = 0; i < cands.size(); ++i) {
            const auto& c = cands[i];
            if (!c.name_norm.empty()) idx_name_exact[c.name_norm].push_back(i);
            if (!c.addr_norm.empty()) idx_addr_exact[c.addr_norm].push_back(i);
            for (const auto& t : c.name_toks) idx_name_tok[t].push_back(i);
            for (const auto& num : c.nums) idx_num[num].push_back(i);
        }
    }

    std::vector<uint32_t> query_candidates(const Record& q) const {
        std::unordered_set<uint32_t> seen;
        auto add_hits = [&](const auto& map, const std::string& key, size_t max_hits = 300) {
            auto it = map.find(key);
            if (it != map.end() && it->second.size() <= max_hits) {
                for (uint32_t idx : it->second) {
                    seen.insert(idx);
                    if (seen.size() >= MAX_CAND_PER_QUERY) return;
                }
            }
        };

        if (!q.name_norm.empty()) add_hits(idx_name_exact, q.name_norm, 100);
        if (!q.addr_norm.empty()) add_hits(idx_addr_exact, q.addr_norm, 100);

        for (const auto& t : q.name_toks) {
            add_hits(idx_name_tok, t, 60);
            if (seen.size() >= MAX_CAND_PER_QUERY) break;
        }
        for (const auto& num : q.nums) {
            add_hits(idx_num, num, 40);
            if (seen.size() >= MAX_CAND_PER_QUERY) break;
        }

        std::vector<uint32_t> res(seen.begin(), seen.end());
        return res;
    }
};

struct MatchTriple {
    float conf;
    std::string s1_id;
    std::string cand_id;
};

std::vector<Record> load_tsv(const std::string& path, int src_filter, const std::string& country_filter) {
    std::vector<Record> records;
    std::ifstream file(path);
    if (!file.is_open()) return records;

    std::string line;
    std::getline(file, line); // header

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        size_t p1 = line.find('\t');
        if (p1 == std::string::npos) continue;
        size_t p2 = line.find('\t', p1 + 1);
        if (p2 == std::string::npos) continue;
        size_t p3 = line.find('\t', p2 + 1);
        if (p3 == std::string::npos) continue;

        std::string id = line.substr(0, p1);
        int src = (id.rfind("S1-", 0) == 0) ? 1 : ((id.rfind("S2-", 0) == 0) ? 2 : 3);
        if (src_filter > 0 && src != src_filter) continue;

        std::string country = line.substr(p3 + 1);
        while (!country.empty() && (country.back() == '\r' || country.back() == ' ' || country.back() == '\n')) country.pop_back();
        if (!country_filter.empty() && country != country_filter) continue;

        std::string raw_name = line.substr(p1 + 1, p2 - p1 - 1);
        std::string raw_addr = line.substr(p2 + 1, p3 - p2 - 1);

        Record r;
        r.id = id;
        r.src = src;
        r.country = country;
        r.is_us = (country == "US");
        r.name_norm = clean_str(raw_name);
        r.addr_norm = clean_str(raw_addr);
        r.addr_missing = r.addr_norm.empty();

        r.name_toks = tokenize(r.name_norm, &NAME_STOPWORDS);
        r.addr_toks = tokenize(r.addr_norm, &ADDR_STOPWORDS);
        r.nums = extract_numbers(r.addr_norm);

        records.push_back(std::move(r));
    }
    return records;
}

int main(int argc, char** argv) {
    std::cout << "======================================================================" << std::endl;
    std::cout << "AMAZON ML CHALLENGE 2026 — HIGH PERFORMANCE C++17 OPENMP ENGINE" << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::string data_dir = (argc > 1) ? argv[1] : "dataset/student_resource/dataset/test";
    std::string out_dir = (argc > 2) ? argv[2] : "output";

    std::string s1_path = data_dir + "/test_source1.tsv";
    std::string s2_path = data_dir + "/test_source2.tsv";
    std::string s3_path = data_dir + "/test_source3.tsv";

    std::ofstream f_match(out_dir + "/matching_results.tsv");
    std::ofstream f_cand(out_dir + "/candidate_pairs.tsv");
    f_match << "source1_entity_id\tmatched_entity_ids\n";
    f_cand << "source1_entity_id\tcandidate_entity_ids\n";

    std::vector<std::string> countries = {"France", "India", "US"};
    auto t_start = std::chrono::high_resolution_clock::now();

    for (const auto& country : countries) {
        std::cout << "\n>>> Loading data for " << country << "..." << std::endl;
        auto raw_queries = load_tsv(s1_path, 1, country);
        auto cands_s2 = load_tsv(s2_path, 2, country);
        auto cands_s3 = load_tsv(s3_path, 3, country);

        std::vector<Record> pool_records;
        pool_records.reserve(cands_s2.size() + cands_s3.size());
        for (auto& r : cands_s2) pool_records.push_back(std::move(r));
        for (auto& r : cands_s3) pool_records.push_back(std::move(r));

        std::cout << "Building candidate pool (" << pool_records.size() << " candidates)..." << std::endl;
        CandidatePool pool;
        pool.build(std::move(pool_records));

        // Precompute n-grams only for query records
        std::cout << "Preparing " << raw_queries.size() << " queries..." << std::endl;
        std::vector<QueryRecord> queries;
        queries.reserve(raw_queries.size());
        for (auto& q : raw_queries) {
            QueryRecord qr;
            qr.name_c2 = get_char_ngrams(q.name_norm, 2);
            qr.name_c3 = get_char_ngrams(q.name_norm, 3);
            if (!q.addr_missing) {
                qr.addr_c2 = get_char_ngrams(q.addr_norm, 2);
                qr.addr_c3 = get_char_ngrams(q.addr_norm, 3);
            }
            qr.base = std::move(q);
            queries.push_back(std::move(qr));
        }

        std::cout << "Parallel inference over " << queries.size() << " queries..." << std::endl;
        std::vector<std::vector<MatchTriple>> thread_triples(omp_get_max_threads());
        std::vector<std::string> cand_lines(queries.size());

        #pragma omp parallel for schedule(dynamic, 1000)
        for (size_t i = 0; i < queries.size(); ++i) {
            int tid = omp_get_thread_num();
            const auto& qr = queries[i];
            const auto& q = qr.base;
            auto cand_indices = pool.query_candidates(q);

            std::string cand_str;
            for (size_t k = 0; k < cand_indices.size(); ++k) {
                if (k > 0) cand_str += ",";
                cand_str += pool.cands[cand_indices[k]].id;
            }
            cand_lines[i] = q.id + "\t" + cand_str;

            for (uint32_t c_idx : cand_indices) {
                const auto& c = pool.cands[c_idx];

                // Name intersections
                size_t n_inter = 0;
                for (const auto& t : q.name_toks) {
                    if (std::find(c.name_toks.begin(), c.name_toks.end(), t) != c.name_toks.end()) n_inter++;
                }
                size_t n_union = q.name_toks.size() + c.name_toks.size() - n_inter;
                float name_tok_jac = (n_union > 0) ? (float)n_inter / (float)n_union : 0.0f;
                float name_contain_q = (!q.name_toks.empty()) ? (float)n_inter / (float)q.name_toks.size() : 0.0f;
                float name_contain_c = (!c.name_toks.empty()) ? (float)n_inter / (float)c.name_toks.size() : 0.0f;
                float name_eq = (!q.name_norm.empty() && q.name_norm == c.name_norm) ? 1.0f : 0.0f;

                // Addr intersections
                size_t a_inter = 0;
                for (const auto& t : q.addr_toks) {
                    if (std::find(c.addr_toks.begin(), c.addr_toks.end(), t) != c.addr_toks.end()) a_inter++;
                }
                size_t a_union = q.addr_toks.size() + c.addr_toks.size() - a_inter;
                float addr_tok_jac = (a_union > 0) ? (float)a_inter / (float)a_union : 0.0f;
                float addr_contain_q = (!q.addr_toks.empty()) ? (float)a_inter / (float)q.addr_toks.size() : 0.0f;
                float addr_contain_c = (!c.addr_toks.empty()) ? (float)a_inter / (float)c.addr_toks.size() : 0.0f;
                float addr_eq = (!q.addr_norm.empty() && q.addr_norm == c.addr_norm) ? 1.0f : 0.0f;

                // Nums
                size_t num_ov = 0;
                for (const auto& n : q.nums) {
                    if (std::find(c.nums.begin(), c.nums.end(), n) != c.nums.end()) num_ov++;
                }

                // Lazy N-Gram Dice calculation on retrieved candidates
                auto c_name_c2 = get_char_ngrams(c.name_norm, 2);
                float name_c2 = dice_sim(qr.name_c2, c_name_c2);
                auto c_name_c3 = get_char_ngrams(c.name_norm, 3);
                float name_c3 = dice_sim(qr.name_c3, c_name_c3);

                auto c_addr_c2 = (!c.addr_missing) ? get_char_ngrams(c.addr_norm, 2) : std::vector<uint64_t>();
                float addr_c2 = dice_sim(qr.addr_c2, c_addr_c2);
                auto c_addr_c3 = (!c.addr_missing) ? get_char_ngrams(c.addr_norm, 3) : std::vector<uint64_t>();
                float addr_c3 = dice_sim(qr.addr_c3, c_addr_c3);

                // Rule evaluation
                float rule_conf = 0.0f;
                if (name_eq > 0.5f && (addr_eq > 0.5f || addr_tok_jac >= 0.35f || num_ov > 0)) {
                    rule_conf = 0.95f + 0.04f * std::min(1.0f, addr_tok_jac + (num_ov > 0 ? 0.05f : 0.0f));
                } else if (addr_eq > 0.5f && (name_tok_jac >= 0.30f || name_c2 >= 0.40f)) {
                    rule_conf = 0.94f + 0.05f * std::min(1.0f, name_tok_jac);
                } else if ((name_tok_jac >= 0.65f || name_c3 >= 0.70f) && (addr_tok_jac >= 0.45f || addr_c2 >= 0.60f)) {
                    rule_conf = 0.92f + 0.05f * std::min(1.0f, (name_tok_jac + addr_tok_jac) / 2.0f);
                } else if (num_ov > 0 && (name_tok_jac >= 0.45f || name_c2 >= 0.55f) && (addr_tok_jac >= 0.30f || addr_c2 >= 0.45f)) {
                    rule_conf = 0.91f + 0.05f * std::min(1.0f, (name_tok_jac + addr_tok_jac) / 2.0f);
                } else if ((addr_tok_jac >= 0.50f || addr_c3 >= 0.65f) && num_ov > 0 && name_c2 >= 0.30f) {
                    rule_conf = 0.85f + 0.05f * addr_tok_jac;
                } else if ((name_tok_jac >= 0.55f || name_c3 >= 0.60f) && (addr_tok_jac >= 0.35f || addr_c2 >= 0.50f)) {
                    rule_conf = 0.80f + 0.05f * name_tok_jac;
                }

                if (rule_conf > 0.0f) {
                    float f[19];
                    f[0] = name_tok_jac;
                    f[1] = name_contain_q;
                    f[2] = name_contain_c;
                    f[3] = name_c2;
                    f[4] = name_c3;
                    f[5] = addr_tok_jac;
                    f[6] = addr_contain_q;
                    f[7] = addr_contain_c;
                    f[8] = addr_c2;
                    f[9] = addr_c3;
                    f[10] = (float)num_ov;
                    f[11] = (float)n_inter;
                    f[12] = name_eq;
                    f[13] = addr_eq;
                    f[14] = (name_eq > 0.5f && addr_eq > 0.5f) ? 1.0f : 0.0f;
                    f[15] = (name_tok_jac > 0.99f) ? 1.0f : 0.0f;
                    f[16] = (addr_tok_jac > 0.99f) ? 1.0f : 0.0f;
                    f[17] = (q.addr_missing || c.addr_missing) ? 1.0f : 0.0f;
                    f[18] = q.is_us ? 1.0f : 0.0f;

                    float p = predict_lgbm_prob(f);
                    float hybrid_conf = std::sqrt(p * rule_conf);
                    if (hybrid_conf >= ANCHOR_THRESH) {
                        thread_triples[tid].push_back({hybrid_conf, q.id, c.id});
                    }
                }
            }
        }

        for (const auto& line : cand_lines) {
            f_cand << line << "\n";
        }

        std::vector<MatchTriple> all_triples;
        for (auto& t_vec : thread_triples) {
            all_triples.insert(all_triples.end(), t_vec.begin(), t_vec.end());
        }

        std::sort(all_triples.begin(), all_triples.end(), [](const MatchTriple& a, const MatchTriple& b) {
            return a.conf > b.conf;
        });

        std::unordered_set<std::string> assigned_candidates;
        std::unordered_map<std::string, std::vector<std::string>> s1_matches;

        for (const auto& t : all_triples) {
            if (assigned_candidates.count(t.cand_id)) continue;
            auto& m_list = s1_matches[t.s1_id];
            if (m_list.size() >= MAX_MATCHES_PER_S1) continue;
            m_list.push_back(t.cand_id);
            assigned_candidates.insert(t.cand_id);
        }

        for (const auto& qr : queries) {
            const auto& q = qr.base;
            auto it = s1_matches.find(q.id);
            if (it != s1_matches.end()) {
                std::string match_str;
                for (size_t k = 0; k < it->second.size(); ++k) {
                    if (k > 0) match_str += ",";
                    match_str += it->second[k];
                }
                f_match << q.id << "\t" << match_str << "\n";
            } else {
                f_match << q.id << "\t\n";
            }
        }

        std::cout << country << " processed successfully." << std::endl;
    }

    f_match.close();
    f_cand.close();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "\nALL COUNTRIES COMPLETED IN " << elapsed << " SECONDS!" << std::endl;
    return 0;
}
