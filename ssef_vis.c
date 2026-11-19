#define _GNU_SOURCE
#include <arpa/inet.h>
#include <ctype.h>
#include <dlfcn.h>
#include <math.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include "rknn_api.h"

/* Set to 1 to match cv2.flip(frame, 1) on PC webcam; set to 0 if PC did not flip */
#define MIRROR_SELFIE 1

#define CAM_W 640
#define CAM_H 480
#define CAM_BYTES (CAM_W * CAM_H * 3)
#define VIS_W 320
#define VIS_H 240
#define VIS_RGB_BYTES (VIS_W * VIS_H * 3)
#define BMP_HDR_SIZE 54
#define BMP_TOTAL_SIZE (BMP_HDR_SIZE + VIS_RGB_BYTES)

#define DET_SIZE 192
#define NUM_ANCHORS 2016
#define LMK_SIZE 224
#define NUM_CLASSES 38
#define VOTE_WINDOW 6
#define VOTE_THRESH 0.52f

static volatile int g_running = 1;
static void on_sigint(int sig) { (void)sig; g_running = 0; }

static const char *LABELS[NUM_CLASSES] = {
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "L", "M", "O", "P", "Q", "R", "S", "T", "U", "V",
    "Y", "Z", "K", "N", "W", "X", "0", "HELLO", "BYE"
};

static pthread_mutex_t g_vis_mtx = PTHREAD_MUTEX_INITIALIZER;
static uint8_t g_vis_bmp[BMP_TOTAL_SIZE];
static uint64_t g_vis_seq = 0;

static void init_bmp_header(uint8_t *hdr, int w, int h) {
    memset(hdr, 0, BMP_HDR_SIZE);
    uint32_t img_sz = (uint32_t)(w * h * 3);
    uint32_t file_sz = BMP_HDR_SIZE + img_sz;
    hdr[0] = 'B'; hdr[1] = 'M';
    memcpy(hdr + 2, &file_sz, 4);
    uint32_t off = BMP_HDR_SIZE;
    memcpy(hdr + 10, &off, 4);
    uint32_t dib = 40;
    memcpy(hdr + 14, &dib, 4);
    int32_t bw = w, bh = -h;
    memcpy(hdr + 18, &bw, 4);
    memcpy(hdr + 22, &bh, 4);
    uint16_t planes = 1, bpp = 24;
    memcpy(hdr + 26, &planes, 2);
    memcpy(hdr + 28, &bpp, 2);
    memcpy(hdr + 34, &img_sz, 4);
}

static void draw_px(uint8_t *bgr, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || x >= VIS_W || y < 0 || y >= VIS_H) return;
    uint8_t *p = bgr + (y * VIS_W + x) * 3;
    p[0] = b; p[1] = g; p[2] = r;
}

static void draw_dot(uint8_t *bgr, int cx, int cy, int rad, uint8_t r, uint8_t g, uint8_t b) {
    for (int dy = -rad; dy <= rad; dy++)
        for (int dx = -rad; dx <= rad; dx++)
            if (dx * dx + dy * dy <= rad * rad)
                draw_px(bgr, cx + dx, cy + dy, r, g, b);
}

static void draw_line(uint8_t *bgr, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;) {
        draw_px(bgr, x0, y0, r, g, b);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

static void draw_rect(uint8_t *bgr, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b) {
    draw_line(bgr, x0, y0, x1, y0, r, g, b);
    draw_line(bgr, x1, y0, x1, y1, r, g, b);
    draw_line(bgr, x1, y1, x0, y1, r, g, b);
    draw_line(bgr, x0, y1, x0, y0, r, g, b);
}

typedef struct { char ch; uint8_t col[5]; } Glyph;
static const Glyph FONT5X7[] = {
    {' ', {0x00,0x00,0x00,0x00,0x00}}, {'-', {0x08,0x08,0x08,0x08,0x08}},
    {'.', {0x00,0x60,0x60,0x00,0x00}}, {':', {0x00,0x36,0x36,0x00,0x00}},
    {'%', {0x23,0x13,0x08,0x64,0x62}}, {'=', {0x14,0x14,0x14,0x14,0x14}},
    {'+', {0x08,0x08,0x3E,0x08,0x08}}, {'/', {0x20,0x10,0x08,0x04,0x02}},
    {'(', {0x00,0x1C,0x22,0x41,0x00}}, {')', {0x00,0x41,0x22,0x1C,0x00}},
    {'[', {0x00,0x7F,0x41,0x41,0x00}}, {']', {0x00,0x41,0x41,0x7F,0x00}},
    {'0', {0x3E,0x51,0x49,0x45,0x3E}}, {'1', {0x00,0x42,0x7F,0x40,0x00}},
    {'2', {0x42,0x61,0x51,0x49,0x46}}, {'3', {0x21,0x41,0x45,0x4B,0x31}},
    {'4', {0x18,0x14,0x12,0x7F,0x10}}, {'5', {0x27,0x45,0x45,0x45,0x39}},
    {'6', {0x3C,0x4A,0x49,0x49,0x30}}, {'7', {0x01,0x71,0x09,0x05,0x03}},
    {'8', {0x36,0x49,0x49,0x49,0x36}}, {'9', {0x06,0x49,0x49,0x29,0x1E}},
    {'A', {0x7E,0x11,0x11,0x11,0x7E}}, {'B', {0x7F,0x49,0x49,0x49,0x36}},
    {'C', {0x3E,0x41,0x41,0x41,0x22}}, {'D', {0x7F,0x41,0x41,0x22,0x1C}},
    {'E', {0x7F,0x49,0x49,0x49,0x41}}, {'F', {0x7F,0x09,0x09,0x09,0x01}},
    {'G', {0x3E,0x41,0x49,0x49,0x7A}}, {'H', {0x7F,0x08,0x08,0x08,0x7F}},
    {'I', {0x00,0x41,0x7F,0x41,0x00}}, {'J', {0x20,0x40,0x41,0x3F,0x01}},
    {'K', {0x7F,0x08,0x14,0x22,0x41}}, {'L', {0x7F,0x40,0x40,0x40,0x40}},
    {'M', {0x7F,0x02,0x0C,0x02,0x7F}}, {'N', {0x7F,0x04,0x08,0x10,0x7F}},
    {'O', {0x3E,0x41,0x41,0x41,0x3E}}, {'P', {0x7F,0x09,0x09,0x09,0x06}},
    {'Q', {0x3E,0x41,0x51,0x21,0x5E}}, {'R', {0x7F,0x09,0x19,0x29,0x46}},
    {'S', {0x46,0x49,0x49,0x49,0x31}}, {'T', {0x01,0x01,0x7F,0x01,0x01}},
    {'U', {0x3F,0x40,0x40,0x40,0x3F}}, {'V', {0x1F,0x20,0x40,0x20,0x1F}},
    {'W', {0x3F,0x40,0x38,0x40,0x3F}}, {'X', {0x63,0x14,0x08,0x14,0x63}},
    {'Y', {0x07,0x08,0x70,0x08,0x07}}, {'Z', {0x61,0x51,0x49,0x45,0x43}},
};

static void draw_text(uint8_t *bgr, int x0, int y0, const char *txt, int scale, uint8_t r, uint8_t g, uint8_t b) {
    int cx = x0;
    for (size_t i = 0; txt[i]; i++) {
        char ch = (char)toupper((unsigned char)txt[i]);
        const uint8_t *cols = FONT5X7[0].col;
        for (size_t g_i = 0; g_i < sizeof(FONT5X7)/sizeof(FONT5X7[0]); g_i++)
            if (FONT5X7[g_i].ch == ch) { cols = FONT5X7[g_i].col; break; }
        for (int c = 0; c < 5; c++)
            for (int row = 0; row < 7; row++)
                if (cols[c] & (1 << row))
                    for (int dy = 0; dy < scale; dy++)
                        for (int dx = 0; dx < scale; dx++)
                            draw_px(bgr, cx + c * scale + dx, y0 + row * scale + dy, r, g, b);
        cx += 6 * scale;
    }
}

static const int HAND_EDGES[21][2] = {
    {0,1},{1,2},{2,3},{3,4},
    {0,5},{5,6},{6,7},{7,8},
    {5,9},{9,10},{10,11},{11,12},
    {9,13},{13,14},{14,15},{15,16},
    {13,17},{0,17},{17,18},{18,19},{19,20}
};

static void *http_server_thread(void *arg) {
    (void)arg;
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) return NULL;
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(8080);

    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("HTTP bind :8080"); close(srv); return NULL;
    }
    listen(srv, 4);
    printf("[VIS] Open in your Windows browser: http://172.32.0.93:8080\n");

    uint8_t *local_bmp = malloc(BMP_TOTAL_SIZE);
    while (g_running) {
        int cli = accept(srv, NULL, NULL);
        if (cli < 0) continue;
        char req[512] = {0};
        recv(cli, req, sizeof(req) - 1, 0);

        if (strncmp(req, "GET /stream", 11) == 0) {
            const char *hdr =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                "Cache-Control: no-cache\r\n\r\n";
            if (send(cli, hdr, strlen(hdr), MSG_NOSIGNAL) <= 0) { close(cli); continue; }

            uint64_t last_seq = 0;
            while (g_running) {
                pthread_mutex_lock(&g_vis_mtx);
                uint64_t cur_seq = g_vis_seq;
                if (cur_seq != last_seq) memcpy(local_bmp, g_vis_bmp, BMP_TOTAL_SIZE);
                pthread_mutex_unlock(&g_vis_mtx);

                if (cur_seq == last_seq) { usleep(15000); continue; }
                last_seq = cur_seq;

                char part[128];
                int pn = snprintf(part, sizeof(part),
                    "--frame\r\nContent-Type: image/bmp\r\nContent-Length: %d\r\n\r\n",
                    BMP_TOTAL_SIZE);
                if (send(cli, part, (size_t)pn, MSG_NOSIGNAL) <= 0) break;
                if (send(cli, local_bmp, BMP_TOTAL_SIZE, MSG_NOSIGNAL) <= 0) break;
                if (send(cli, "\r\n", 2, MSG_NOSIGNAL) <= 0) break;
            }
        } else {
            const char *html =
                "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
                "<!DOCTYPE html><html><body style='background:#111;color:#eee;font-family:sans-serif;text-align:center'>"
                "<h2>Luckfox RV1106 PC-Aligned Dual-Hand ASL Pipeline</h2>"
                "<img src='/stream' style='width:640px;height:480px;border:2px solid #444;image-rendering:pixelated'/>"
                "<p>Cyan = Right Hand [R] | Magenta = Left Hand [L] | Yellow = MediaPipe ROI</p>"
                "</body></html>";
            send(cli, html, strlen(html), MSG_NOSIGNAL);
        }
        close(cli);
    }
    free(local_bmp);
    close(srv);
    return NULL;
}

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
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return -1; }
    fclose(f);
    int ret = rknn_init(&m->ctx, buf, (uint32_t)sz, 0, NULL);
    free(buf);
    if (ret != RKNN_SUCC) return -1;

    rknn_query(m->ctx, RKNN_QUERY_IN_OUT_NUM, &m->io_num, sizeof(m->io_num));
    for (uint32_t i = 0; i < m->io_num.n_input; i++) {
        m->in_attr[i].index = i;
        rknn_query(m->ctx, RKNN_QUERY_INPUT_ATTR, &m->in_attr[i], sizeof(rknn_tensor_attr));
        uint32_t msz = m->in_attr[i].size_with_stride > m->in_attr[i].size ? m->in_attr[i].size_with_stride : m->in_attr[i].size;
        m->in_mem[i] = rknn_create_mem(m->ctx, msz);
        m->in_attr[i].pass_through = 1;
        rknn_set_io_mem(m->ctx, m->in_mem[i], &m->in_attr[i]);
    }
    for (uint32_t i = 0; i < m->io_num.n_output; i++) {
        m->out_attr[i].index = i;
        rknn_query(m->ctx, RKNN_QUERY_OUTPUT_ATTR, &m->out_attr[i], sizeof(rknn_tensor_attr));
        uint32_t msz = m->out_attr[i].size_with_stride > m->out_attr[i].size ? m->out_attr[i].size_with_stride : m->out_attr[i].size;
        m->out_mem[i] = rknn_create_mem(m->ctx, msz);
        m->out_attr[i].pass_through = 1;
        rknn_set_io_mem(m->ctx, m->out_mem[i], &m->out_attr[i]);
    }
    return 0;
}

static int find_out_by_elems(const RknnModel *m, uint32_t elems, int occurrence) {
    int seen = 0;
    for (uint32_t i = 0; i < m->io_num.n_output; i++) {
        if (m->out_attr[i].n_elems == elems) {
            if (seen == occurrence) return (int)i;
            seen++;
        }
    }
    return -1;
}

static int find_out_by_name_sub(const RknnModel *m, const char *sub) {
    for (uint32_t i = 0; i < m->io_num.n_output; i++)
        if (strstr(m->out_attr[i].name, sub) != NULL) return (int)i;
    return -1;
}

static inline float dequant_i8(int8_t v, const rknn_tensor_attr *attr) {
    return ((float)v - (float)attr->zp) * attr->scale;
}
static inline int8_t quant_i8(float v, const rknn_tensor_attr *attr) {
    int q = (int)roundf(v / attr->scale) + attr->zp;
    return (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
}
static inline float to_prob(float raw) {
    if (raw >= 0.0f && raw <= 1.0f) return raw;
    return 1.0f / (1.0f + expf(-raw));
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
typedef struct { float score, cx, cy, w, h, kp0x, kp0y, kp2x, kp2y; } PalmBox;

static void mirror_frame_640x480(uint8_t *buf) {
    for (int y = 0; y < CAM_H; y++) {
        uint8_t *row = buf + y * CAM_W * 3;
        for (int x = 0; x < CAM_W / 2; x++) {
            int rx = CAM_W - 1 - x;
            uint8_t r = row[x * 3 + 0], g = row[x * 3 + 1], b = row[x * 3 + 2];
            row[x * 3 + 0] = row[rx * 3 + 0];
            row[x * 3 + 1] = row[rx * 3 + 1];
            row[x * 3 + 2] = row[rx * 3 + 2];
            row[rx * 3 + 0] = r;
            row[rx * 3 + 1] = g;
            row[rx * 3 + 2] = b;
        }
    }
}

static void letterbox_640x480_to_192_i8(const uint8_t *src, int8_t *dst, int32_t zp) {
    memset(dst, (int8_t)zp, DET_SIZE * DET_SIZE * 3);
    for (int dy = 0; dy < 144; dy++) {
        int sy = (dy * CAM_H) / 144;
        const uint8_t *srow = src + sy * CAM_W * 3;
        int8_t *drow = dst + (dy + 24) * DET_SIZE * 3;
        for (int dx = 0; dx < DET_SIZE; dx++) {
            int sx = (dx * CAM_W) / DET_SIZE;
            int r = (int)srow[sx * 3 + 0] - 128 + zp;
            int g = (int)srow[sx * 3 + 1] - 128 + zp;
            int b = (int)srow[sx * 3 + 2] - 128 + zp;
            drow[dx * 3 + 0] = (int8_t)(r > 127 ? 127 : (r < -128 ? -128 : r));
            drow[dx * 3 + 1] = (int8_t)(g > 127 ? 127 : (g < -128 ? -128 : g));
            drow[dx * 3 + 2] = (int8_t)(b > 127 ? 127 : (b < -128 ? -128 : b));
        }
    }
}

static int collect_top_palms(const int8_t *reg_i8, const rknn_tensor_attr *reg_attr,
                             const int8_t *cls_i8, const rknn_tensor_attr *cls_attr,
                             PalmBox out[5], float *max_score_out) {
    PalmBox cands[5] = {0};
    float max_sc = 0.0f;
    for (int i = 0; i < NUM_ANCHORS; i++) {
        float logit = dequant_i8(cls_i8[i], cls_attr);
        float score = 1.0f / (1.0f + expf(-logit));
        if (score > max_sc) max_sc = score;
        if (score < 0.14f) continue;

        const int8_t *r = reg_i8 + i * 18;
        float w = dequant_i8(r[2], reg_attr) / (float)DET_SIZE;
        float h = dequant_i8(r[3], reg_attr) / (float)DET_SIZE;
        if (w < 0.05f || h < 0.05f || w > 0.85f || h > 0.85f) continue;
        float aspect = w / h;
        if (aspect < 0.45f || aspect > 2.2f) continue;

        float cx = g_anchors[i][0] + dequant_i8(r[0], reg_attr) / (float)DET_SIZE;
        float cy = g_anchors[i][1] + dequant_i8(r[1], reg_attr) / (float)DET_SIZE;

        int dup = -1;
        for (int k = 0; k < 5; k++) {
            if (cands[k].score > 0.0f && fabsf(cands[k].cx - cx) < 0.09f && fabsf(cands[k].cy - cy) < 0.09f) {
                dup = k; break;
            }
        }
        if (dup >= 0) {
            if (score <= cands[dup].score) continue;
            cands[dup].score = 0.0f;
        }

        int worst = 0;
        for (int k = 1; k < 5; k++)
            if (cands[k].score < cands[worst].score) worst = k;
        if (score > cands[worst].score) {
            cands[worst].score = score;
            cands[worst].cx = cx; cands[worst].cy = cy;
            cands[worst].w  = w;  cands[worst].h  = h;
            cands[worst].kp0x = g_anchors[i][0] + dequant_i8(r[4], reg_attr) / (float)DET_SIZE;
            cands[worst].kp0y = g_anchors[i][1] + dequant_i8(r[5], reg_attr) / (float)DET_SIZE;
            cands[worst].kp2x = g_anchors[i][0] + dequant_i8(r[8], reg_attr) / (float)DET_SIZE;
            cands[worst].kp2y = g_anchors[i][1] + dequant_i8(r[9], reg_attr) / (float)DET_SIZE;
        }
    }
    for (int a = 0; a < 4; a++)
        for (int b = a + 1; b < 5; b++)
            if (cands[b].score > cands[a].score) { PalmBox t = cands[a]; cands[a] = cands[b]; cands[b] = t; }
    int n = 0;
    for (int k = 0; k < 5; k++) if (cands[k].score >= 0.14f) out[n++] = cands[k];
    *max_score_out = max_sc;
    return n;
}

/* Official MediaPipe palm_detection_detection_to_roi.pbtxt: shift_y = -0.5, scale = 2.6 */
static void palm_box_to_rect(const PalmBox *box, float *cx_px, float *cy_px, float *box_sz, float *rot) {
    float x = box->cx * CAM_W;
    float y = ((box->cy - 0.125f) / 0.75f) * CAM_H;
    float bw = box->w * CAM_W;
    float bh = (box->h / 0.75f) * CAM_H;
    float k0x = box->kp0x * CAM_W, k0y = ((box->kp0y - 0.125f) / 0.75f) * CAM_H;
    float k2x = box->kp2x * CAM_W, k2y = ((box->kp2y - 0.125f) / 0.75f) * CAM_H;

    float vx = k2x - k0x, vy = k2y - k0y;
    float kp_dist = hypotf(vx, vy);
    float r = (kp_dist > 8.0f) ? (atan2f(vy, vx) + (float)M_PI * 0.5f) : 0.0f;

    float base_span = bw > bh ? bw : bh;
    x += 0.50f * base_span * sinf(r);
    y -= 0.50f * base_span * cosf(r);

    float sz = base_span * 2.60f;
    if (sz < 160.0f) sz = 160.0f;
    if (sz > 420.0f) sz = 420.0f;

    *cx_px = x; *cy_px = y; *box_sz = sz; *rot = r;
}

/* Official MediaPipe hand_landmark_landmarks_to_roi.pbtxt: shift_y = -0.1, scale = 2.0 */
static void landmarks_to_rect(const float pts[21][3], float *cx_px, float *cy_px, float *box_sz, float *rot) {
    float wx = pts[0][0] * CAM_W, wy = pts[0][1] * CAM_H;
    float mx = pts[9][0] * CAM_W, my = pts[9][1] * CAM_H;
    float target_r = atan2f(my - wy, mx - wx) + (float)M_PI * 0.5f;

    float diff_r = target_r - (*rot);
    while (diff_r >  (float)M_PI) diff_r -= 2.0f * (float)M_PI;
    while (diff_r < -(float)M_PI) diff_r += 2.0f * (float)M_PI;
    float r = (*rot) + 0.75f * diff_r;
    float cr = cosf(r), sr = sinf(r);

    float min_ux = 1e9f, max_ux = -1e9f, min_uy = 1e9f, max_uy = -1e9f;
    for (int p = 0; p < 21; p++) {
        float dx = pts[p][0] * CAM_W - wx;
        float dy = pts[p][1] * CAM_H - wy;
        float ux =  dx * cr + dy * sr;
        float uy = -dx * sr + dy * cr;
        if (ux < min_ux) min_ux = ux;
        if (ux > max_ux) max_ux = ux;
        if (uy < min_uy) min_uy = uy;
        if (uy > max_uy) max_uy = uy;
    }

    float mid_ux = 0.5f * (min_ux + max_ux);
    float mid_uy = 0.5f * (min_uy + max_uy);
    float span_x = max_ux - min_ux;
    float span_y = max_uy - min_uy;
    float max_span = (span_x > span_y) ? span_x : span_y;

    float palm_len = hypotf(mx - wx, my - wy);
    if (max_span < palm_len * 1.18f) max_span = palm_len * 1.18f;

    mid_uy -= 0.10f * max_span;

    float target_cx = wx + mid_ux * cr - mid_uy * sr;
    float target_cy = wy + mid_ux * sr + mid_uy * cr;
    float target_sz = max_span * 2.00f;
    if (target_sz < 150.0f) target_sz = 150.0f;
    if (target_sz > 420.0f) target_sz = 420.0f;

    *cx_px  = 0.80f * target_cx + 0.20f * (*cx_px);
    *cy_px  = 0.80f * target_cy + 0.20f * (*cy_px);
    *box_sz = 0.70f * target_sz + 0.30f * (*box_sz);
    *rot    = r;
}

static int landmarks_plausible(const float pts[21][3]) {
    float wx = pts[0][0] * CAM_W, wy = pts[0][1] * CAM_H;
    float mx = pts[9][0] * CAM_W, my = pts[9][1] * CAM_H;
    float palm_len = hypotf(mx - wx, my - wy);
    if (palm_len < 22.0f || palm_len > 250.0f) return 0;

    float ix = pts[5][0] * CAM_W, iy = pts[5][1] * CAM_H;
    float px = pts[17][0] * CAM_W, py = pts[17][1] * CAM_H;
    float palm_wid = hypotf(ix - px, iy - py);
    if (palm_wid < palm_len * 0.30f || palm_wid > palm_len * 1.65f) return 0;

    int good_fingers = 0;
    const int mcp[4] = {5, 9, 13, 17};
    for (int f = 0; f < 4; f++) {
        int m = mcp[f], p = m + 1;
        float d = hypotf((pts[p][0] - pts[m][0]) * CAM_W, (pts[p][1] - pts[m][1]) * CAM_H);
        if (d >= palm_len * 0.15f) good_fingers++;
    }
    if (good_fingers < 3) return 0;

    int in_bounds = 0;
    for (int p = 0; p < 21; p++) {
        if (pts[p][0] >= -0.04f && pts[p][0] <= 1.04f &&
            pts[p][1] >= -0.04f && pts[p][1] <= 1.04f) in_bounds++;
    }
    return in_bounds >= 17;
}

/* Anatomically accurate handedness in screen coordinates (+x right, +y down):
 * On a mirrored selfie frame, Right hand palm-facing has Index(5) left of Pinky(17) -> cross > 0 */
static float infer_handedness(const float pts[21][3], float model_handed) {
    float ux = (pts[5][0] - pts[0][0]) * CAM_W;
    float uy = (pts[5][1] - pts[0][1]) * CAM_H;
    float vx = (pts[17][0] - pts[0][0]) * CAM_W;
    float vy = (pts[17][1] - pts[0][1]) * CAM_H;
    float norm = hypotf(ux, uy) * hypotf(vx, vy) + 1e-6f;
    float sin_palm = (ux * vy - uy * vx) / norm;

    float dz_fingers = ((pts[8][2] - pts[5][2]) + (pts[12][2] - pts[9][2]) +
                        (pts[16][2] - pts[13][2]) + (pts[20][2] - pts[17][2])) * 0.25f;
    float dz_thumb = pts[4][2] - pts[5][2];
    if (dz_fingers > 0.025f && dz_thumb > 0.015f) {
        sin_palm = -sin_palm;
    }

#if !MIRROR_SELFIE
    sin_palm = -sin_palm;
#endif

    if (fabsf(sin_palm) < 0.14f) return model_handed;
    float geom_right = (sin_palm > 0.0f) ? 1.0f : 0.0f;
    return 0.80f * geom_right + 0.20f * model_handed;
}

/* Bilinear crop interpolation matching MediaPipe ImageToTensorCalculator */
static void crop_hand_roi_i8(const uint8_t *src, float cx_px, float cy_px, float box_sz, float rot,
                             int8_t *dst, int32_t zp) {
    float cos_r = cosf(rot), sin_r = sinf(rot);
    float inv_sz = box_sz / (float)LMK_SIZE;
    for (int y = 0; y < LMK_SIZE; y++) {
        float ny = ((float)y + 0.5f - (float)LMK_SIZE * 0.5f) * inv_sz;
        for (int x = 0; x < LMK_SIZE; x++) {
            float nx = ((float)x + 0.5f - (float)LMK_SIZE * 0.5f) * inv_sz;
            float fx = cx_px + nx * cos_r - ny * sin_r;
            float fy = cy_px + nx * sin_r + ny * cos_r;
            int8_t *dp = dst + (y * LMK_SIZE + x) * 3;
            int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
            if (x0 >= 0 && x0 < CAM_W - 1 && y0 >= 0 && y0 < CAM_H - 1) {
                int wx = (int)((fx - (float)x0) * 256.0f);
                int wy = (int)((fy - (float)y0) * 256.0f);
                int iwx = 256 - wx, iwy = 256 - wy;
                int w00 = (iwx * iwy) >> 8, w10 = (wx * iwy) >> 8;
                int w01 = (iwx * wy)  >> 8, w11 = (wx * wy)  >> 8;
                const uint8_t *p00 = src + (y0 * CAM_W + x0) * 3;
                const uint8_t *p10 = p00 + 3;
                const uint8_t *p01 = p00 + CAM_W * 3;
                const uint8_t *p11 = p01 + 3;
                for (int c = 0; c < 3; c++) {
                    int val = (p00[c] * w00 + p10[c] * w10 + p01[c] * w01 + p11[c] * w11) >> 8;
                    int q = val - 128 + zp;
                    dp[c] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
                }
            } else {
                dp[0] = (int8_t)zp; dp[1] = (int8_t)zp; dp[2] = (int8_t)zp;
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

typedef struct { float prob_hist[VOTE_WINDOW][NUM_CLASSES]; int count, head; } SoftVoteWindow;
static void vote_reset(SoftVoteWindow *vw) { vw->count = 0; vw->head = 0; }
static int vote_push_logits(SoftVoteWindow *vw, const int8_t *logits_i8, const rknn_tensor_attr *attr,
                            int *raw_cls_out, int *second_cls_out, float *top1_pct_out) {
    float logits[NUM_CLASSES], max_l = -1e9f;
    int best_raw = 0, sec_raw = 1;
    for (int c = 0; c < NUM_CLASSES; c++) {
        logits[c] = dequant_i8(logits_i8[c], attr);
        if (logits[c] > max_l) { max_l = logits[c]; best_raw = c; }
    }
    float sec_l = -1e9f;
    for (int c = 0; c < NUM_CLASSES; c++) {
        if (c != best_raw && logits[c] > sec_l) { sec_l = logits[c]; sec_raw = c; }
    }
    float sum_e = 0.0f;
    for (int c = 0; c < NUM_CLASSES; c++) {
        float e = expf(logits[c] - max_l);
        vw->prob_hist[vw->head][c] = e;
        sum_e += e;
    }
    for (int c = 0; c < NUM_CLASSES; c++) vw->prob_hist[vw->head][c] /= sum_e;

    vw->head = (vw->head + 1) % VOTE_WINDOW;
    if (vw->count < VOTE_WINDOW) vw->count++;

    int best_c = -1;
    float best_avg = 0.0f;
    for (int c = 0; c < NUM_CLASSES; c++) {
        float acc = 0.0f;
        for (int i = 0; i < vw->count; i++) acc += vw->prob_hist[i][c];
        float avg = acc / (float)vw->count;
        if (avg > best_avg) { best_avg = avg; best_c = c; }
    }
    *raw_cls_out = best_raw;
    *second_cls_out = sec_raw;
    *top1_pct_out = best_avg * 100.0f;
    return (vw->count >= 3 && best_avg >= VOTE_THRESH) ? best_c : -1;
}

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void run_landmark_roi(RknnModel *lmk_m, const uint8_t *cam_buf,
                             float cx, float cy, float sz, float rot,
                             int lmk_pts_idx, int lmk_pres_idx, int lmk_hand_idx,
                             float pts_out[21][3], float *pres_out, float *hand_out) {
    int8_t *lmk_in_buf    = (int8_t *)lmk_m->in_mem[0]->virt_addr;
    const int8_t *pts_i8  = (const int8_t *)lmk_m->out_mem[lmk_pts_idx]->virt_addr;
    const int8_t *pres_i8 = (const int8_t *)lmk_m->out_mem[lmk_pres_idx]->virt_addr;
    const int8_t *hand_i8 = (const int8_t *)lmk_m->out_mem[lmk_hand_idx]->virt_addr;

    crop_hand_roi_i8(cam_buf, cx, cy, sz, rot, lmk_in_buf, lmk_m->in_attr[0].zp);
    rknn_run(lmk_m->ctx, NULL);

    *pres_out = to_prob(dequant_i8(pres_i8[0], &lmk_m->out_attr[lmk_pres_idx]));
    *hand_out = to_prob(dequant_i8(hand_i8[0], &lmk_m->out_attr[lmk_hand_idx]));

    float cr = cosf(rot), sr = sinf(rot);
    for (int p = 0; p < 21; p++) {
        float lx = (dequant_i8(pts_i8[p * 3 + 0], &lmk_m->out_attr[lmk_pts_idx]) - 112.0f) * (sz / 224.0f);
        float ly = (dequant_i8(pts_i8[p * 3 + 1], &lmk_m->out_attr[lmk_pts_idx]) - 112.0f) * (sz / 224.0f);
        float lz =  dequant_i8(pts_i8[p * 3 + 2], &lmk_m->out_attr[lmk_pts_idx]) * (sz / 224.0f);
        pts_out[p][0] = (cx + lx * cr - ly * sr) / (float)CAM_W;
        pts_out[p][1] = (cy + lx * sr + ly * cr) / (float)CAM_H;
        pts_out[p][2] = lz / (float)CAM_W;
    }
}

typedef struct {
    int active;
    int lost_frames;
    int valid_this_frame;
    float cx, cy, sz, rot;
    float smooth_pts[21][3];
    float ema_handed;
    int is_right;
    float last_pres;
} HandTrack;

int main(void) {
    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);
    build_palm_anchors();
    init_bmp_header(g_vis_bmp, VIS_W, VIS_H);

    pthread_t http_tid;
    pthread_create(&http_tid, NULL, http_server_thread, NULL);

    void *cam_lib = dlopen("/userdata/libssef_capture_rgb.so", RTLD_NOW);
    if (!cam_lib) return 1;
    int (*ssef_rgb_init)(void) = dlsym(cam_lib, "ssef_rgb_init");
    int (*ssef_rgb_get_frame)(uint8_t *) = dlsym(cam_lib, "ssef_rgb_get_frame");
    void (*ssef_rgb_close)(void) = dlsym(cam_lib, "ssef_rgb_close");

    RknnModel det_m, lmk_m, mlp_m;
    if (load_rknn_model("/userdata/hand_detector.rknn", &det_m) != 0) return 1;
    if (load_rknn_model("/userdata/hand_landmarks_detector.rknn", &lmk_m) != 0) return 1;
    if (load_rknn_model("/userdata/landmark_mlp.rknn", &mlp_m) != 0) return 1;

    int det_reg_idx = find_out_by_elems(&det_m, NUM_ANCHORS * 18, 0);
    int det_cls_idx = find_out_by_elems(&det_m, NUM_ANCHORS, 0);
    int lmk_pts_idx  = find_out_by_elems(&lmk_m, 63, 0);
    int lmk_pres_idx = find_out_by_name_sub(&lmk_m, "Identity_1");
    int lmk_hand_idx = find_out_by_name_sub(&lmk_m, "Identity_2");
    if (lmk_pres_idx < 0) lmk_pres_idx = find_out_by_elems(&lmk_m, 1, 0);
    if (lmk_hand_idx < 0) lmk_hand_idx = find_out_by_elems(&lmk_m, 1, 1);

    if (ssef_rgb_init() != 0) return 1;

    uint8_t *cam_buf = malloc(CAM_BYTES);
    uint8_t *canvas  = malloc(VIS_RGB_BYTES);
    int8_t *det_in_buf   = (int8_t *)det_m.in_mem[0]->virt_addr;
    const int8_t *reg_i8 = (const int8_t *)det_m.out_mem[det_reg_idx]->virt_addr;
    const int8_t *cls_i8 = (const int8_t *)det_m.out_mem[det_cls_idx]->virt_addr;

    int8_t *mlp_in_buf       = (int8_t *)mlp_m.in_mem[0]->virt_addr;
    const int8_t *mlp_out_i8 = (const int8_t *)mlp_m.out_mem[0]->virt_addr;

    SoftVoteWindow vw;
    vote_reset(&vw);
    int frame_cnt = 0, total_frames = 0;
    double fps_t0 = now_ms(), cur_fps = 0.0;

    HandTrack tracks[2] = {{0}};

    while (g_running) {
        if (ssef_rgb_get_frame(cam_buf) != CAM_BYTES) { usleep(5000); continue; }
        total_frames++;

#if MIRROR_SELFIE
        mirror_frame_640x480(cam_buf);
#endif

        for (int y = 0; y < VIS_H; y++) {
            const uint8_t *srow = cam_buf + (y * 2) * CAM_W * 3;
            uint8_t *drow = canvas + y * VIS_W * 3;
            for (int x = 0; x < VIS_W; x++) {
                const uint8_t *sp = srow + (x * 2) * 3;
                drow[x * 3 + 0] = sp[2];
                drow[x * 3 + 1] = sp[1];
                drow[x * 3 + 2] = sp[0];
            }
        }

        int active_count = 0;
        for (int t = 0; t < 2; t++) {
            tracks[t].valid_this_frame = 0;
            if (!tracks[t].active) continue;

            float pts[21][3], pres = 0.0f, model_hand = 0.5f;
            float probe_sz = (tracks[t].lost_frames > 0) ? (tracks[t].sz * 1.25f) : tracks[t].sz;
            run_landmark_roi(&lmk_m, cam_buf, tracks[t].cx, tracks[t].cy, probe_sz, tracks[t].rot,
                             lmk_pts_idx, lmk_pres_idx, lmk_hand_idx, pts, &pres, &model_hand);
            tracks[t].last_pres = pres;

            if (pres >= 0.32f && landmarks_plausible(pts)) {
                tracks[t].valid_this_frame = 1;
                tracks[t].lost_frames = 0;
                for (int p = 0; p < 21; p++)
                    for (int c = 0; c < 3; c++)
                        tracks[t].smooth_pts[p][c] = 0.72f * pts[p][c] + 0.28f * tracks[t].smooth_pts[p][c];

                float h_est = infer_handedness(tracks[t].smooth_pts, model_hand);
                tracks[t].ema_handed = 0.85f * tracks[t].ema_handed + 0.15f * h_est;
                if (tracks[t].ema_handed >= 0.62f) tracks[t].is_right = 1;
                else if (tracks[t].ema_handed <= 0.38f) tracks[t].is_right = 0;

                landmarks_to_rect(tracks[t].smooth_pts, &tracks[t].cx, &tracks[t].cy, &tracks[t].sz, &tracks[t].rot);
                active_count++;
            } else {
                tracks[t].lost_frames++;
                if (tracks[t].lost_frames >= 4) {
                    tracks[t].active = 0;
                } else {
                    active_count++;
                }
            }
        }

        if (tracks[0].active && tracks[1].active) {
            float dx = (tracks[0].smooth_pts[0][0] - tracks[1].smooth_pts[0][0]) * CAM_W;
            float dy = (tracks[0].smooth_pts[0][1] - tracks[1].smooth_pts[0][1]) * CAM_H;
            if (hypotf(dx, dy) < 60.0f) {
                if (tracks[0].lost_frames <= tracks[1].lost_frames) tracks[1].active = 0;
                else tracks[0].active = 0;
                active_count = 1;
            }
        }

        float max_palm_score = 0.0f;
        int should_search = (active_count == 0) || (active_count == 1 && (total_frames % 2 == 0));
        if (should_search) {
            letterbox_640x480_to_192_i8(cam_buf, det_in_buf, det_m.in_attr[0].zp);
            rknn_run(det_m.ctx, NULL);
            PalmBox cands[5];
            int nc = collect_top_palms(reg_i8, &det_m.out_attr[det_reg_idx],
                                       cls_i8, &det_m.out_attr[det_cls_idx],
                                       cands, &max_palm_score);
            for (int k = 0; k < nc; k++) {
                int free_slot = -1;
                for (int t = 0; t < 2; t++) if (!tracks[t].active) { free_slot = t; break; }
                if (free_slot < 0) break;

                float rx, ry, rsz, rrot;
                palm_box_to_rect(&cands[k], &rx, &ry, &rsz, &rrot);

                int overlaps = 0;
                for (int t = 0; t < 2; t++) {
                    if (!tracks[t].active) continue;
                    float d_center = hypotf(rx - tracks[t].cx, ry - tracks[t].cy);
                    float d_wrist  = hypotf(rx - tracks[t].smooth_pts[0][0] * CAM_W,
                                            ry - tracks[t].smooth_pts[0][1] * CAM_H);
                    if (d_center < 90.0f || d_wrist < 90.0f) { overlaps = 1; break; }
                }
                if (overlaps) continue;

                float pts[21][3], pres = 0.0f, model_hand = 0.5f;
                run_landmark_roi(&lmk_m, cam_buf, rx, ry, rsz, rrot,
                                 lmk_pts_idx, lmk_pres_idx, lmk_hand_idx, pts, &pres, &model_hand);
                if (pres >= 0.32f && landmarks_plausible(pts)) {
                    int bx1 = (int)((cands[k].cx - 0.5f * cands[k].w) * VIS_W);
                    int by1 = (int)(((cands[k].cy - 0.5f * cands[k].h - 0.125f) / 0.75f) * VIS_H);
                    int bx2 = (int)((cands[k].cx + 0.5f * cands[k].w) * VIS_W);
                    int by2 = (int)(((cands[k].cy + 0.5f * cands[k].h - 0.125f) / 0.75f) * VIS_H);
                    draw_rect(canvas, bx1, by1, bx2, by2, 0, 255, 0);

                    HandTrack *nt = &tracks[free_slot];
                    memset(nt, 0, sizeof(*nt));
                    nt->active = 1;
                    nt->valid_this_frame = 1;
                    nt->cx = rx; nt->cy = ry; nt->sz = rsz; nt->rot = rrot;
                    nt->last_pres = pres;
                    memcpy(nt->smooth_pts, pts, sizeof(pts));
                    float h_est = infer_handedness(pts, model_hand);
                    nt->ema_handed = h_est;
                    nt->is_right = (h_est >= 0.50f) ? 1 : 0;
                    landmarks_to_rect(nt->smooth_pts, &nt->cx, &nt->cy, &nt->sz, &nt->rot);
                }
            }
        }

        float feat[131] = {0};
        int has_left = 0, has_right = 0;
        float lw[3] = {0}, rw[3] = {0};

        for (int t = 0; t < 2; t++) {
            if (!tracks[t].active || !tracks[t].valid_this_frame) continue;

            float half = tracks[t].sz * 0.5f, cr = cosf(tracks[t].rot), sr = sinf(tracks[t].rot);
            float cx_s[4] = {-half, half, half, -half}, cy_s[4] = {-half, -half, half, half};
            int rx[4], ry[4];
            for (int c = 0; c < 4; c++) {
                rx[c] = (int)((tracks[t].cx + cx_s[c] * cr - cy_s[c] * sr) * 0.5f);
                ry[c] = (int)((tracks[t].cy + cx_s[c] * sr + cy_s[c] * cr) * 0.5f);
            }
            for (int c = 0; c < 4; c++)
                draw_line(canvas, rx[c], ry[c], rx[(c + 1) & 3], ry[(c + 1) & 3], 255, 220, 0);

            uint8_t er = tracks[t].is_right ? 0   : 255;
            uint8_t eg = tracks[t].is_right ? 220 : 60;
            uint8_t eb = 255;
            for (int e = 0; e < 21; e++) {
                int p0 = HAND_EDGES[e][0], p1 = HAND_EDGES[e][1];
                draw_line(canvas,
                    (int)(tracks[t].smooth_pts[p0][0] * VIS_W), (int)(tracks[t].smooth_pts[p0][1] * VIS_H),
                    (int)(tracks[t].smooth_pts[p1][0] * VIS_W), (int)(tracks[t].smooth_pts[p1][1] * VIS_H),
                    er, eg, eb);
            }
            for (int p = 0; p < 21; p++)
                draw_dot(canvas, (int)(tracks[t].smooth_pts[p][0] * VIS_W),
                                 (int)(tracks[t].smooth_pts[p][1] * VIS_H), 2, 255, 40, 40);

            float scale = 0.0f;
            if (!tracks[t].is_right && !has_left) {
                normalize_hand_63(tracks[t].smooth_pts, &feat[0], &scale);
                feat[126] = 1.0f;
                has_left = 1;
                memcpy(lw, tracks[t].smooth_pts[0], sizeof(lw));
            } else if (tracks[t].is_right && !has_right) {
                normalize_hand_63(tracks[t].smooth_pts, &feat[63], &scale);
                feat[127] = 1.0f;
                has_right = 1;
                memcpy(rw, tracks[t].smooth_pts[0], sizeof(rw));
            }
        }

        char status_line[64];
        if (!has_left && !has_right) {
            int holding = (tracks[0].active || tracks[1].active);
            if (!holding) vote_reset(&vw);
            snprintf(status_line, sizeof(status_line), "FPS:%.0f %s P:%.2f",
                     cur_fps, holding ? "HOLD" : "SEARCH", max_palm_score);
        } else {
            if (has_left && has_right) {
                feat[128] = rw[0] - lw[0];
                feat[129] = rw[1] - lw[1];
                feat[130] = rw[2] - lw[2];
            }

            for (int j = 0; j < 131; j++)
                mlp_in_buf[j] = quant_i8(feat[j], &mlp_m.in_attr[0]);
            rknn_run(mlp_m.ctx, NULL);

            int raw_cls = 0, sec_cls = 1;
            float top1_pct = 0.0f;
            int stable_cls = vote_push_logits(&vw, mlp_out_i8, &mlp_m.out_attr[0], &raw_cls, &sec_cls, &top1_pct);
            const char *htag = (has_left && has_right) ? "L+R" : (has_right ? "R" : "L");
            snprintf(status_line, sizeof(status_line), "FPS:%.0f [%s] %s/%s->%s",
                     cur_fps, htag, LABELS[raw_cls], LABELS[sec_cls],
                     stable_cls >= 0 ? LABELS[stable_cls] : "...");
        }

        if (total_frames % 20 == 0) {
            printf("[F %4d] %s\n", total_frames, status_line);
            fflush(stdout);
        }

        for (int y = 0; y < 22; y++)
            memset(canvas + y * VIS_W * 3, 15, VIS_W * 3);
        draw_text(canvas, 6, 4, status_line, 2, 255, 255, 0);

        pthread_mutex_lock(&g_vis_mtx);
        memcpy(g_vis_bmp + BMP_HDR_SIZE, canvas, VIS_RGB_BYTES);
        g_vis_seq++;
        pthread_mutex_unlock(&g_vis_mtx);

        frame_cnt++;
        double now = now_ms();
        if (now - fps_t0 >= 1000.0) {
            cur_fps = (frame_cnt * 1000.0) / (now - fps_t0);
            frame_cnt = 0;
            fps_t0 = now;
        }
    }

    ssef_rgb_close();
    free(cam_buf); free(canvas);
    return 0;
}
