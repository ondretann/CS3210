#include "kseq/kseq.h"
#include "common.h"
#include <iostream>

struct KSeqChar {
    const char* seq;
    const char* qual;
    int len;
};

__device__ double calc_match_confidence(const KSeqChar* sample, int start, int sig_len) {
    double score = 0.0;
    const char* qual = sample->qual;
    for (int i = 0; i < sig_len; i++) {
        score += qual[start + i] - 33;
    }
    return score / sig_len;
}

__device__ double sample_sign_match(const KSeqChar* sample, const KSeqChar* signature) {
    int samp_len = sample->len;
    int sig_len = signature->len;
    const char* samp_seq = sample->seq;
    const char* sig_seq = signature->seq;

    if (samp_len >= sig_len) {
        // Sliding window over the sample sequence
        for (int i = 0; i <= samp_len - sig_len; i++) {
            bool match = true;
            for (int j = 0; j < sig_len; j++) {
                if (samp_seq[i + j] == 'N' || sig_seq[j] == 'N') {
                    continue;
                }

                if (samp_seq[i + j] != sig_seq[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return calc_match_confidence(sample, i, sig_len);
            }
        }
    }
    return -1.0; 
}

__global__ void matcher_kernel(KSeqChar* samples, KSeqChar* signatures, double* con_scores, int num_samples, int num_signatures) {
    int sample_idx = blockIdx.x;
    int signature_idx = threadIdx.x;

    double match_score = sample_sign_match(&samples[sample_idx], &signatures[signature_idx]);

    // Write the result to the appropriate position in the result array
    if (match_score >= 0.0) {
        int result_index = sample_idx * num_signatures + signature_idx;
        con_scores[result_index] = match_score;
    }
}

void runMatcher(const std::vector<klibpp::KSeq>& samples, const std::vector<klibpp::KSeq>& signatures, std::vector<MatchResult>& matches) {
    // preprocessing begin
    int num_samples = samples.size();
    int num_signatures = signatures.size();

    KSeqChar* kseq_sample = new KSeqChar[num_samples];
    KSeqChar* kseq_sign = new KSeqChar[num_signatures];

    for (int i = 0; i < num_samples; i++) {
        kseq_sample[i].len = samples[i].seq.size();

        char* d_seq;
        cudaMalloc((void**)&d_seq, samples[i].seq.size() + 1);
        cudaMemcpy(d_seq, samples[i].seq.c_str(), samples[i].seq.size() + 1, cudaMemcpyHostToDevice);
        kseq_sample[i].seq = d_seq;

        char* d_qual;
        cudaMalloc((void**)&d_qual, samples[i].qual.size() + 1);
        cudaMemcpy(d_qual, samples[i].qual.c_str(), samples[i].qual.size() + 1, cudaMemcpyHostToDevice);
        kseq_sample[i].qual = d_qual;
    }

    for (int i = 0; i < num_signatures; i++) {
        kseq_sign[i].len = signatures[i].seq.size();

        char* d_seq;
        cudaMalloc((void**)&d_seq, signatures[i].seq.size() + 1);
        cudaMemcpy(d_seq, signatures[i].seq.c_str(), signatures[i].seq.size() + 1, cudaMemcpyHostToDevice);
        kseq_sign[i].seq = d_seq;
    }

    KSeqChar* d_samples;
    KSeqChar* d_signatures;
    cudaMalloc(&d_samples, num_samples * sizeof(KSeqChar));
    cudaMalloc(&d_signatures, num_signatures * sizeof(KSeqChar));

    cudaMemcpy(d_samples, kseq_sample, num_samples * sizeof(KSeqChar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_signatures, kseq_sign, num_signatures * sizeof(KSeqChar), cudaMemcpyHostToDevice);

    int array_size = num_samples * num_signatures;
    double* d_con_scores;
    cudaMalloc(&d_con_scores, array_size * sizeof(double));
    cudaMemset(d_con_scores, -1, array_size * sizeof(double));

    dim3 gridDim(num_samples);
    dim3 blockDim(num_signatures);

    matcher_kernel<<<gridDim, blockDim>>>(d_samples, d_signatures, d_con_scores, num_samples, num_signatures);
    cudaDeviceSynchronize();
    
    double* h_con_scores = new double[array_size];
    cudaMemcpy(h_con_scores, d_con_scores, array_size * sizeof(double), cudaMemcpyDeviceToHost);

    for (int k = 0; k < array_size; k++) {
        if (h_con_scores[k] >= 0.0) {
            MatchResult result;
            int samp_index = k / num_signatures;
            int sign_index = k % num_signatures;
            result.sample_name = samples[samp_index].name;
            result.signature_name = signatures[sign_index].name;
            result.match_score = h_con_scores[k];
            matches.push_back(result);
        }
    }

    delete[] kseq_sample;
    delete[] kseq_sign;
    delete[] h_con_scores;

    cudaFree(d_samples);
    cudaFree(d_signatures);
    cudaFree(d_con_scores);
}
