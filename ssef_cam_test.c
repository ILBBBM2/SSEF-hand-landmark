#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include "rknn_api.h"

#define CAM_W 640
#define CAM_H 480
#define CAM_BYTES (CAM_W * CAM_H * 3)
#define DET_SIZE 192
#define NUM_ANCHORS 2016
#define LMK_SIZE 224
#define NUM_CLASSES 38
#define VOTE_WINDOW 8
#define VOTE_THRESH 0.70f

static volatile int g_running = 1;
static void on_sigint(int sig) { (void)sig; g_running = 0; }

static const char *LABELS[NUM_CLASSES] = {
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "L", "M", "O", "P", "Q", "R", "S", "T", "U", "V",
    "Y", "Z", "K", "N", "W", "X", "0", "HELLO", "BYE"
};

typedef struct {
    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr in_attr[4];
    rknn_tensor_attr out_attr[8];
    rknn_tensor_mem *in_mem[4];
    rknn_tensor_mem *out_mem[8];
} RknnModel;

static int load_rknn_model(const char *path, RknnModel *m) {
    memset(m, 0, sizeof(*m));
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    void *buf = malloc((size_t)sz);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f); free(buf); return -1;
    }
    fclose(f);

    int ret = rknn_init(&m->ctx, buf, (uint32_t)sz, 0, NULL);
    free(buf);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "rknn_init(%s) failed: %d\n", path, ret);
        return -1;
    }

    rknn_query(m->ctx, RKNN_QUERY_IN_OUT_NUM, &m->io_num, sizeof(m->io_num));
    for (uint32_t i = 0; i < m->io_num.n_input; i++) {
        m->in_attr[i].index = i;
        rknn_query(m->ctx, RKNN_QUERY_INPUT_ATTR, &m->in_attr[i], sizeof(rknn_tensor_attr));
        uint32_t msz = m->in_attr[i].size_with_stride > m->in_attr[i].size
                       ? m->in_attr[i].size_with_stride : m->in_attr[i].size;
        m->in_mem[i] = rknn_create_mem(m->ctx, msz);
        if (!m->in_mem[i]) { fprintf(stderr, "rknn_create_mem in failed\n"); return -1; }
        m->in_attr[i].pass_through = 1;
        ret = rknn_set_io_mem(m->ctx, m->in_mem[i], &m->in_attr[i]);
        if (ret != RKNN_SUCC) { fprintf(stderr, "rknn_set_io_mem in failed: %d\n", ret); return -1; }
    }
    for (uint32_t i = 0; i < m->io_num.n_output; i++) {
        m->out_attr[i].index = i;
        rknn_query(m->ctx, RKNN_QUERY_OUTPUT_ATTR, &m->out_attr[i], sizeof(rknn_tensor_attr));
        uint32_t msz = m->out_attr[i].size_with_stride > m->out_attr[i].size
                       ? m->out_attr[i].size_with_stride : m->out_attr[i].size;
        m->out_mem[i] = rknn_create_mem(m->ctx, msz);
        if (!m->out_mem[i]) { fprintf(stderr, "rknn_create_mem out failed\n"); return -1; }
        m->out_attr[i].pass_through = 1;
        ret = rknn_set_io_mem(m->ctx, m->out_mem[i], &m->out_attr[i]);
        if (ret != RKNN_SUCC) { fprintf(stderr, "rknn_set_io_mem out failed: %d\n", ret); return -1; }
    }
    return 0;
}

static void destroy_rknn_model(RknnModel *m) {
    for (uint32_t i = 0; i < m->io_num.n_input; i++)
        if (m->in_mem[i]) rknn_destroy_mem(m->ctx, m->in_mem[i]);
    for (uint32_t i = 0; i < m->io_num.n_output; i++)
        if (m->out_mem[i]) rknn_destroy_mem(m->ctx, m->out_mem[i]);
    if (m->ctx) rknn_destroy(m->ctx);
}

static inline float dequant_i8(int8_t v, const rknn_tensor_attr *attr) {
    return ((float)v - (float)attr->zp) * attr->scale;
}

static inline int8_t quant_i8(float v, const rknn_tensor_attr *attr) {
    int q = (int)roundf(v / attr->scale) + attr->zp;
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

static float g_anchors[NUM_ANCHORS][2];
static void build_palm_anchors(void) {
    int idx = 0;
    for (int y = 0; y < 24; y++)
        for (int x = 0; x < 24; x++)
            for (int k = 0; k < 2; k++) {
                g_anchors[idx][0] = (x + 0.5f) / 24.0f;
                g_anchors[idx][1] = (y + 0.5f) / 24.0f;
                idx++;
            }
    for (int y = 0; y < 12; y++)
        for (int x = 0; x < 12; x++)
            for (int k = 0; k < 6; k++) {
                g_anchors[idx][0] = (x + 0.5f) / 12.0f;
                g_anchors[idx][1] = (y + 0.5f) / 12.0f;
                idx++;
            }
}

typedef struct {
    float score, cx, cy, w, h, kp0x, kp0y, kp2x, kp2y;
} PalmBox;

static float box_iou(const PalmBox *a, const PalmBox *b) {
    float ax1 = a->cx - 0.5f * a->w, ay1 = a->cy - 0.5f * a->h;
    float ax2 = a->cx + 0.5f * a->w, ay2 = a->cy + 0.5f * a->h;
    float bx1 = b->cx - 0.5f * b->w, by1 = b->cy - 0.5f * b->h;
    float bx2 = b->cx + 0.5f * b->w, by2 = b->cy + 0.5f * b->h;
    float ix1 = ax1 > bx1 ? ax1 : bx1, iy1 = ay1 > by1 ? ay1 : by1;
    float ix2 = ax2 < bx2 ? ax2 : bx2, iy2 = ay2 < by2 ? ay2 : by2;
    float iw = ix2 - ix1, ih = iy2 - iy1;
    if (iw <= 0.0f || ih <= 0.0f) return 0.0f;
    float inter = iw * ih;
    float uni = a->w * a->h + b->w * b->h - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

/* Letterbox 640x480 uint8 RGB -> 192x192 int8 NHWC (zp = -128 -> uint8 ^ 0x80) */
static void letterbox_640x480_to_192_i8(const uint8_t *src, int8_t *dst) {
    memset(dst, -128, DET_SIZE * DET_SIZE * 3);
    for (int dy = 0; dy < 144; dy++) {
        int sy = (dy * CAM_H) / 144;
        const uint8_t *srow = src + sy * CAM_W * 3;
        int8_t *drow = dst + (dy + 24) * DET_SIZE * 3;
        for (int dx = 0; dx < DET_SIZE; dx++) {
            int sx = (dx * CAM_W) / DET_SIZE;
            drow[dx * 3 + 0] = (int8_t)(srow[sx * 3 + 2] ^ 0x80);
            drow[dx * 3 + 1] = (int8_t)(srow[sx * 3 + 1] ^ 0x80);
            drow[dx * 3 + 2] = (int8_t)(srow[sx * 3 + 0] ^ 0x80);
        }
    }
}

/* Extract rotated/scaled 224x224 hand ROI directly into int8 NPU buffer (zp = -128) */
static void crop_hand_roi_i8(const uint8_t *src, const PalmBox *box, int8_t *dst,
                             float *out_cx_px, float *out_cy_px, float *out_box_sz, float *out_rot) {
    float cx_px = box->cx * CAM_W;
    float cy_px = ((box->cy - 0.125f) / 0.75f) * CAM_H;
    float bw_px = box->w * CAM_W;
    float bh_px = (box->h / 0.75f) * CAM_H;
    float k0x = box->kp0x * CAM_W, k0y = ((box->kp0y - 0.125f) / 0.75f) * CAM_H;
    float k2x = box->kp2x * CAM_W, k2y = ((box->kp2y - 0.125f) / 0.75f) * CAM_H;

    float vx = k2x - k0x, vy = k2y - k0y;
    float rot = atan2f(vy, vx) + (float)M_PI * 0.5f;
    float box_sz = (bw_px > bh_px ? bw_px : bh_px) * 2.6f;
    if (box_sz < 64.0f) box_sz = 64.0f;

    float dir_x = sinf(rot), dir_y = -cosf(rot);
    cx_px += 0.5f * box_sz * dir_x;
    cy_px += 0.5f * box_sz * dir_y;

    *out_cx_px = cx_px; *out_cy_px = cy_px; *out_box_sz = box_sz; *out_rot = rot;

    float cos_r = cosf(rot), sin_r = sinf(rot);
    float inv_sz = box_sz / (float)LMK_SIZE;
    for (int y = 0; y < LMK_SIZE; y++) {
        float ny = ((float)y + 0.5f - (float)LMK_SIZE * 0.5f) * inv_sz;
        for (int x = 0; x < LMK_SIZE; x++) {
            float nx = ((float)x + 0.5f - (float)LMK_SIZE * 0.5f) * inv_sz;
            int sx = (int)(cx_px + nx * cos_r - ny * sin_r + 0.5f);
            int sy = (int)(cy_px + nx * sin_r + ny * cos_r + 0.5f);
            int8_t *dp = dst + (y * LMK_SIZE + x) * 3;
            if (sx >= 0 && sx < CAM_W && sy >= 0 && sy < CAM_H) {
                const uint8_t *sp = src + (sy * CAM_W + sx) * 3;
                dp[0] = (int8_t)(sp[2] ^ 0x80);
                dp[1] = (int8_t)(sp[1] ^ 0x80);
                dp[2] = (int8_t)(sp[0] ^ 0x80);
            } else {
                dp[0] = -128; dp[1] = -128; dp[2] = -128;
            }
        }
    }
}

static void normalize_hand_63(const float raw[21][3], float out63[63], float *palm_scale) {
    float wx = raw[0][0], wy = raw[0][1], wz = raw[0][2];
    float dx = raw[9][0] - wx, dy = raw[9][1] - wy, dz = raw[9][2] - wz;
    float scale = sqrtf(dx * dx + dy * dy + dz * dz);
    if (scale < 1e-6f) scale = 1.0f;
    *palm_scale = scale;
    for (int i = 0; i < 21; i++) {
        out63[i * 3 + 0] = (raw[i][0] - wx) / scale;
        out63[i * 3 + 1] = (raw[i][1] - wy) / scale;
        out63[i * 3 + 2] = (raw[i][2] - wz) / scale;
    }
}

typedef struct { int hist[VOTE_WINDOW], count, head; } VoteWindow;
static void vote_reset(VoteWindow *vw) { vw->count = 0; vw->head = 0; }
static int vote_push(VoteWindow *vw, int cls, float *out_ratio) {
    vw->hist[vw->head] = cls;
    vw->head = (vw->head + 1) % VOTE_WINDOW;
    if (vw->count < VOTE_WINDOW) vw->count++;
    int counts[NUM_CLASSES] = {0};
    for (int i = 0; i < vw->count; i++) {
        int c = vw->hist[i];
        if (c >= 0 && c < NUM_CLASSES) counts[c]++;
    }
    int best_c = -1, best_n = 0;
    for (int c = 0; c < NUM_CLASSES; c++)
        if (counts[c] > best_n) { best_n = counts[c]; best_c = c; }
    *out_ratio = (vw->count > 0) ? ((float)best_n / (float)vw->count) : 0.0f;
    return (vw->count >= 4 && *out_ratio >= VOTE_THRESH) ? best_c : -1;
}

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(void) {
    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);
    build_palm_anchors();

    void *cam_lib = dlopen("/userdata/libssef_capture_rgb.so", RTLD_NOW);
    if (!cam_lib) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }
    int (*ssef_rgb_init)(void) = dlsym(cam_lib, "ssef_rgb_init");
    int (*ssef_rgb_get_frame)(uint8_t *) = dlsym(cam_lib, "ssef_rgb_get_frame");
    void (*ssef_rgb_close)(void) = dlsym(cam_lib, "ssef_rgb_close");

    RknnModel det_m, lmk_m, mlp_m;
    if (load_rknn_model("/userdata/hand_detector.rknn", &det_m) != 0) return 1;
    if (load_rknn_model("/userdata/hand_landmarks_detector.rknn", &lmk_m) != 0) return 1;
    if (load_rknn_model("/userdata/landmark_mlp.rknn", &mlp_m) != 0) return 1;
    printf("[OK] All 3 RKNN models initialized with zero-copy NPU buffers.\n");

    if (ssef_rgb_init() != 0) { fprintf(stderr, "ssef_rgb_init failed\n"); return 1; }
    printf("[OK] Camera stream started (640x480 RGB888). Hold your hand in front of the camera (Ctrl+C to stop)...\n\n");

    uint8_t *cam_buf = malloc(CAM_BYTES);
    int8_t *det_in_buf  = (int8_t *)det_m.in_mem[0]->virt_addr;
    const int8_t *reg_i8 = (const int8_t *)det_m.out_mem[0]->virt_addr;
    const int8_t *cls_i8 = (const int8_t *)det_m.out_mem[1]->virt_addr;

    int8_t *lmk_in_buf  = (int8_t *)lmk_m.in_mem[0]->virt_addr;
    const int8_t *pts_i8  = (const int8_t *)lmk_m.out_mem[0]->virt_addr;
    const int8_t *pres_i8 = (const int8_t *)lmk_m.out_mem[1]->virt_addr;
    const int8_t *hand_i8 = (const int8_t *)lmk_m.out_mem[2]->virt_addr;

    int8_t *mlp_in_buf   = (int8_t *)mlp_m.in_mem[0]->virt_addr;
    const int8_t *mlp_out_i8 = (const int8_t *)mlp_m.out_mem[0]->virt_addr;

    VoteWindow vw;
    vote_reset(&vw);
    int frame_cnt = 0, total_frames = 0;
    double fps_t0 = now_ms(), cur_fps = 0.0;

    while (g_running) {
        double t_start = now_ms();
        if (ssef_rgb_get_frame(cam_buf) != CAM_BYTES) {
            usleep(5000);
            continue;
        }
        total_frames++;

        /* 1. Palm Detector */
        letterbox_640x480_to_192_i8(cam_buf, det_in_buf);
        rknn_run(det_m.ctx, NULL);

        PalmBox cands[64];
        int n_cand = 0;
        float max_palm_score = 0.0f;
        for (int i = 0; i < NUM_ANCHORS; i++) {
            float logit = dequant_i8(cls_i8[i], &det_m.out_attr[1]);
            float score = 1.0f / (1.0f + expf(-logit));
            if (score > max_palm_score) max_palm_score = score;
            if (score < 0.65f) continue;

            const int8_t *r = reg_i8 + i * 18;
            float dx = dequant_i8(r[0], &det_m.out_attr[0]) / (float)DET_SIZE;
            float dy = dequant_i8(r[1], &det_m.out_attr[0]) / (float)DET_SIZE;
            float w  = dequant_i8(r[2], &det_m.out_attr[0]) / (float)DET_SIZE;
            float h  = dequant_i8(r[3], &det_m.out_attr[0]) / (float)DET_SIZE;
            if (w <= 0.01f || h <= 0.01f) continue;

            if (n_cand < 64) {
                PalmBox *b = &cands[n_cand++];
                b->score = score;
                b->cx = g_anchors[i][0] + dx; b->cy = g_anchors[i][1] + dy;
                b->w = w; b->h = h;
                b->kp0x = g_anchors[i][0] + dequant_i8(r[4], &det_m.out_attr[0]) / (float)DET_SIZE;
                b->kp0y = g_anchors[i][1] + dequant_i8(r[5], &det_m.out_attr[0]) / (float)DET_SIZE;
                b->kp2x = g_anchors[i][0] + dequant_i8(r[8], &det_m.out_attr[0]) / (float)DET_SIZE;
                b->kp2y = g_anchors[i][1] + dequant_i8(r[9], &det_m.out_attr[0]) / (float)DET_SIZE;
            }
        }

        for (int i = 0; i < n_cand - 1; i++)
            for (int j = i + 1; j < n_cand; j++)
                if (cands[j].score > cands[i].score) { PalmBox tmp = cands[i]; cands[i] = cands[j]; cands[j] = tmp; }

        PalmBox hands[2];
        int n_hands = 0;
        for (int i = 0; i < n_cand && n_hands < 2; i++) {
            int keep = 1;
            for (int k = 0; k < n_hands; k++)
                if (box_iou(&cands[i], &hands[k]) > 0.30f) { keep = 0; break; }
            if (keep) hands[n_hands++] = cands[i];
        }

        if (n_hands == 0) {
            vote_reset(&vw);
            if (total_frames % 15 == 0) {
                printf("[Frame %4d | FPS %4.1f | %4.1f ms] No hand (max palm score: %.2f, center RGB: %d,%d,%d)\n",
                       total_frames, cur_fps, now_ms() - t_start, max_palm_score,
                       cam_buf[(240 * CAM_W + 320) * 3 + 0],
                       cam_buf[(240 * CAM_W + 320) * 3 + 1],
                       cam_buf[(240 * CAM_W + 320) * 3 + 2]);
                fflush(stdout);
            }
        } else {
            /* 2. Hand Landmarks */
            float hand_pts[2][21][3];
            float hand_side[2] = {0.5f, 0.5f};
            int valid_hands = 0;
            float max_presence = 0.0f;

            for (int h = 0; h < n_hands; h++) {
                float r_cx, r_cy, r_sz, r_rot;
                crop_hand_roi_i8(cam_buf, &hands[h], lmk_in_buf, &r_cx, &r_cy, &r_sz, &r_rot);
                rknn_run(lmk_m.ctx, NULL);

                float raw_p = dequant_i8(pres_i8[0], &lmk_m.out_attr[1]);
                float raw_h = dequant_i8(hand_i8[0], &lmk_m.out_attr[2]);
                float presence = (raw_p >= 0.0f && raw_p <= 1.0f) ? raw_p : (1.0f / (1.0f + expf(-raw_p)));
                float handed   = (raw_h >= 0.0f && raw_h <= 1.0f) ? raw_h : (1.0f / (1.0f + expf(-raw_h)));
                if (presence > max_presence) max_presence = presence;

                if (presence >= 0.5f) {
                    float cos_r = cosf(r_rot), sin_r = sinf(r_rot);
                    for (int p = 0; p < 21; p++) {
                        float lx = (dequant_i8(pts_i8[p * 3 + 0], &lmk_m.out_attr[0]) - 112.0f) * (r_sz / 224.0f);
                        float ly = (dequant_i8(pts_i8[p * 3 + 1], &lmk_m.out_attr[0]) - 112.0f) * (r_sz / 224.0f);
                        float lz =  dequant_i8(pts_i8[p * 3 + 2], &lmk_m.out_attr[0]) * (r_sz / 224.0f);
                        hand_pts[valid_hands][p][0] = (r_cx + lx * cos_r - ly * sin_r) / (float)CAM_W;
                        hand_pts[valid_hands][p][1] = (r_cy + lx * sin_r + ly * cos_r) / (float)CAM_H;
                        hand_pts[valid_hands][p][2] = lz / (float)CAM_W;
                    }
                    hand_side[valid_hands] = handed;
                    valid_hands++;
                }
            }

            if (valid_hands == 0) {
                vote_reset(&vw);
                printf("[Frame %4d | FPS %4.1f] Palm=%.2f but landmark presence=%.2f (<0.50)\n",
                       total_frames, cur_fps, max_palm_score, max_presence);
                fflush(stdout);
            } else {
                /* 3. 131-D Feature Vector + MLP */
                float feat[131] = {0};
                int left_idx = -1, right_idx = -1;
                if (valid_hands == 1) {
                    if (hand_side[0] < 0.5f) left_idx = 0; else right_idx = 0;
                } else {
                    if (hand_side[0] < hand_side[1]) { left_idx = 0; right_idx = 1; }
                    else { left_idx = 1; right_idx = 0; }
                }
                float scale_l = 0.0f, scale_r = 0.0f;
                if (left_idx >= 0) { normalize_hand_63(hand_pts[left_idx], &feat[0], &scale_l); feat[126] = 1.0f; }
                if (right_idx >= 0) { normalize_hand_63(hand_pts[right_idx], &feat[63], &scale_r); feat[127] = 1.0f; }
                if (left_idx >= 0 && right_idx >= 0) {
                    float mean_s = 0.5f * (scale_l + scale_r);
                    if (mean_s < 1e-6f) mean_s = 1.0f;
                    feat[128] = (hand_pts[right_idx][0][0] - hand_pts[left_idx][0][0]) / mean_s;
                    feat[129] = (hand_pts[right_idx][0][1] - hand_pts[left_idx][0][1]) / mean_s;
                    feat[130] = (hand_pts[right_idx][0][2] - hand_pts[left_idx][0][2]) / mean_s;
                }

                for (int j = 0; j < 131; j++) {
                    mlp_in_buf[j] = quant_i8(feat[j], &mlp_m.in_attr[0]);
                }
                rknn_run(mlp_m.ctx, NULL);

                int raw_cls = 0;
                int8_t highest = mlp_out_i8[0];
                for (int c = 1; c < NUM_CLASSES; c++) {
                    if (mlp_out_i8[c] > highest) { highest = mlp_out_i8[c]; raw_cls = c; }
                }

                float vote_ratio = 0.0f;
                int stable_cls = vote_push(&vw, raw_cls, &vote_ratio);
                double lat_ms = now_ms() - t_start;

                printf("[Frame %4d | FPS %4.1f | %4.1f ms] Palm=%.2f Lmk=%.2f (%s) -> Raw: %-8s | Locked: %-8s (%.0f%%)\n",
                       total_frames, cur_fps, lat_ms, max_palm_score, max_presence,
                       (valid_hands == 2) ? "L+R" : (left_idx >= 0 ? "L" : "R"),
                       LABELS[raw_cls],
                       (stable_cls >= 0) ? LABELS[stable_cls] : "...",
                       vote_ratio * 100.0f);
                fflush(stdout);
            }
        }

        frame_cnt++;
        double now = now_ms();
        if (now - fps_t0 >= 1000.0) {
            cur_fps = (frame_cnt * 1000.0) / (now - fps_t0);
            frame_cnt = 0;
            fps_t0 = now;
        }
    }

    printf("\nStopping camera and releasing NPU...\n");
    ssef_rgb_close();
    destroy_rknn_model(&det_m);
    destroy_rknn_model(&lmk_m);
    destroy_rknn_model(&mlp_m);
    free(cam_buf);
    dlclose(cam_lib);
    return 0;
}
